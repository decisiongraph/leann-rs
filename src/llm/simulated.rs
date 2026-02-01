//! Simulated LLM for testing
//!
//! Returns canned responses without requiring external API calls.

/// Simulated LLM provider for testing
pub struct SimulatedLlm {
    model_name: String,
}

impl SimulatedLlm {
    /// Create a new simulated LLM
    pub fn new(model_name: String) -> anyhow::Result<Self> {
        Ok(Self { model_name })
    }

    /// Generate a simulated response
    pub async fn generate(&self, prompt: &str) -> anyhow::Result<String> {
        // Extract the question from the prompt if present
        let question = if prompt.contains("Question:") {
            prompt
                .split("Question:")
                .nth(1)
                .and_then(|s| s.split('\n').next())
                .map(|s| s.trim())
                .unwrap_or("your question")
        } else {
            "your question"
        };

        // Check if there are context passages
        let has_context = prompt.contains("Context:") || prompt.contains("passages");

        let response = if has_context {
            format!(
                "Based on the provided context, here is my response to \"{question}\":\n\n\
                 The information in the documents suggests that this topic is covered in the \
                 retrieved passages. This is a simulated response for testing purposes.\n\n\
                 Key points from the context:\n\
                 1. The first relevant passage discusses the main concepts.\n\
                 2. Additional passages provide supporting information.\n\
                 3. The context contains useful details for answering your query.\n\n\
                 Note: This is a test response from the simulated LLM (model: {}).",
                self.model_name
            )
        } else {
            format!(
                "I understand you're asking about \"{question}\".\n\n\
                 This is a simulated response for testing purposes. In a real scenario, \
                 I would provide a helpful answer based on my training.\n\n\
                 Note: This is a test response from the simulated LLM (model: {}).",
                self.model_name
            )
        };

        Ok(response)
    }
}
