//! Local embeddings using Candle (sentence-transformers compatible)

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use tracing::info;

/// Local embedding provider using Candle
pub struct CandleEmbedding {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimensions: usize,
    normalize: bool,
}

impl CandleEmbedding {
    /// Create a new Candle embedding provider
    ///
    /// Supports models like:
    /// - sentence-transformers/all-MiniLM-L6-v2 (384 dims)
    /// - sentence-transformers/all-mpnet-base-v2 (768 dims)
    /// - BAAI/bge-small-en-v1.5 (384 dims)
    /// - BAAI/bge-base-en-v1.5 (768 dims)
    pub fn new(model_name: String, model_path: Option<String>) -> anyhow::Result<Self> {
        info!("Loading local embedding model: {}", model_name);

        // Determine device (CPU for now, MPS/CUDA can be added later)
        let device = Device::Cpu;

        // Load model files
        let (config_path, tokenizer_path, weights_path) = if let Some(path) = model_path {
            let base = PathBuf::from(path);
            (
                base.join("config.json"),
                base.join("tokenizer.json"),
                base.join("model.safetensors"),
            )
        } else {
            // Download from HuggingFace Hub
            let api = Api::new()?;
            let repo = api.repo(Repo::new(model_name.clone(), RepoType::Model));

            let config = repo.get("config.json")?;
            let tokenizer = repo.get("tokenizer.json")?;

            // Try safetensors first, fall back to pytorch
            let weights = repo
                .get("model.safetensors")
                .or_else(|_| repo.get("pytorch_model.bin"))?;

            (config, tokenizer, weights)
        };

        // Load config
        let config_content = std::fs::read_to_string(&config_path)?;
        let config: BertConfig = serde_json::from_str(&config_content)?;
        let dimensions = config.hidden_size;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load model weights
        let vb = if weights_path.extension().map(|e| e == "safetensors").unwrap_or(false) {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? }
        } else {
            VarBuilder::from_pth(weights_path, DTYPE, &device)?
        };

        let model = BertModel::load(vb, &config)?;

        // Check if model uses normalization (sentence-transformers typically do)
        let normalize = model_name.contains("sentence-transformers")
            || model_name.contains("bge")
            || model_name.contains("e5");

        info!(
            "Loaded model: {} dims, device: {:?}, normalize: {}",
            dimensions, device, normalize
        );

        Ok(Self {
            model,
            tokenizer,
            device,
            dimensions,
            normalize,
        })
    }

    /// Get embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Compute embeddings for texts
    pub fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process in batches to avoid memory issues
        let batch_size = 32;
        for batch in texts.chunks(batch_size) {
            let batch_embeddings = self.embed_batch(batch)?;
            all_embeddings.extend(batch_embeddings);
        }

        Ok(all_embeddings)
    }

    fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        // Get max length for padding
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

        // Create input tensors
        let mut input_ids = Vec::new();
        let mut attention_mask = Vec::new();
        let mut token_type_ids = Vec::new();

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();

            // Pad to max length
            let mut padded_ids = ids.to_vec();
            let mut padded_mask = mask.to_vec();
            let mut padded_types = type_ids.to_vec();

            padded_ids.resize(max_len, 0);
            padded_mask.resize(max_len, 0);
            padded_types.resize(max_len, 0);

            input_ids.extend(padded_ids);
            attention_mask.extend(padded_mask);
            token_type_ids.extend(padded_types);
        }

        let batch_size = encodings.len();

        let input_ids = Tensor::from_vec(input_ids, (batch_size, max_len), &self.device)?
            .to_dtype(DType::U32)?;
        let attention_mask =
            Tensor::from_vec(attention_mask, (batch_size, max_len), &self.device)?
                .to_dtype(DType::U32)?;
        let token_type_ids =
            Tensor::from_vec(token_type_ids, (batch_size, max_len), &self.device)?
                .to_dtype(DType::U32)?;

        // Run model
        let output = self.model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean pooling over sequence length (ignoring padding)
        let embeddings = self.mean_pooling(&output, &attention_mask)?;

        // Optionally normalize
        let embeddings = if self.normalize {
            self.l2_normalize(&embeddings)?
        } else {
            embeddings
        };

        // Convert to Vec<Vec<f32>>
        let embeddings = embeddings.to_dtype(DType::F32)?;
        let data = embeddings.flatten_all()?.to_vec1::<f32>()?;

        let mut result = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * self.dimensions;
            let end = start + self.dimensions;
            result.push(data[start..end].to_vec());
        }

        Ok(result)
    }

    fn mean_pooling(&self, output: &Tensor, attention_mask: &Tensor) -> anyhow::Result<Tensor> {
        // output shape: (batch, seq_len, hidden)
        // attention_mask shape: (batch, seq_len)

        // Expand attention mask to hidden dimension
        let mask = attention_mask
            .to_dtype(output.dtype())?
            .unsqueeze(2)?
            .broadcast_as(output.shape())?;

        // Masked sum
        let masked = output.mul(&mask)?;
        let sum = masked.sum(1)?;

        // Count non-padding tokens
        let count = attention_mask
            .to_dtype(output.dtype())?
            .sum(1)?
            .unsqueeze(1)?
            .broadcast_as(sum.shape())?;

        // Mean
        let mean = sum.div(&count.clamp(1e-9, f64::INFINITY)?)?;

        Ok(mean)
    }

    fn l2_normalize(&self, embeddings: &Tensor) -> anyhow::Result<Tensor> {
        let norm = embeddings
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::INFINITY)?;
        Ok(embeddings.broadcast_div(&norm)?)
    }
}
