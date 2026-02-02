//! Query processing and expansion
//!
//! Index-derived query expansion using BM25 to find related terms.
//! Uses AST-aware patterns to extract meaningful code symbols.

use std::collections::HashMap;

use regex::Regex;
use std::sync::LazyLock;

/// Compiled regex patterns for extracting code symbols
static CODE_SYMBOL_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        // Rust
        Regex::new(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)").unwrap(),
        Regex::new(r"(?:pub\s+)?struct\s+(\w+)").unwrap(),
        Regex::new(r"(?:pub\s+)?enum\s+(\w+)").unwrap(),
        Regex::new(r"(?:pub\s+)?trait\s+(\w+)").unwrap(),
        // Python
        Regex::new(r"(?:async\s+)?def\s+(\w+)").unwrap(),
        Regex::new(r"class\s+(\w+)").unwrap(),
        // JavaScript/TypeScript
        Regex::new(r"(?:async\s+)?function\s+(\w+)").unwrap(),
        Regex::new(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(").unwrap(),
        // Go
        Regex::new(r"func\s+(?:\([^)]+\)\s+)?(\w+)").unwrap(),
        Regex::new(r"type\s+(\w+)\s+(?:struct|interface)").unwrap(),
        // Java/C#
        Regex::new(r"(?:public|private|protected)?\s*(?:static\s+)?(?:class|interface)\s+(\w+)").unwrap(),
    ]
});

/// Extract code symbol names (functions, classes, structs) from text
fn extract_code_symbols(text: &str, max_symbols: usize) -> Vec<String> {
    let mut symbols: HashMap<String, usize> = HashMap::new();

    for pattern in CODE_SYMBOL_PATTERNS.iter() {
        for caps in pattern.captures_iter(text) {
            if let Some(name) = caps.get(1) {
                let name_str = name.as_str().to_string();
                // Skip very short names or test-related names
                if name_str.len() >= 3
                    && !name_str.starts_with("test_")
                    && !name_str.starts_with("_")
                {
                    *symbols.entry(name_str).or_insert(0) += 1;
                }
            }
        }
    }

    // Sort by frequency
    let mut symbol_vec: Vec<_> = symbols.into_iter().collect();
    symbol_vec.sort_by(|a, b| b.1.cmp(&a.1));
    symbol_vec.into_iter().take(max_symbols).map(|(s, _)| s).collect()
}

/// Check if a term looks like code (snake_case, camelCase, or common code patterns)
fn is_code_like(term: &str) -> bool {
    // Contains underscore (snake_case)
    if term.contains('_') {
        return true;
    }
    // Looks like an ID (contains numbers mixed with letters)
    let has_digit = term.chars().any(|c| c.is_ascii_digit());
    let has_letter = term.chars().any(|c| c.is_alphabetic());
    if has_digit && has_letter {
        return true;
    }
    // Common code keywords
    let code_keywords: std::collections::HashSet<&str> = [
        "let", "const", "var", "fn", "func", "def", "pub", "mut", "impl",
        "struct", "enum", "type", "trait", "class", "interface", "async",
        "await", "return", "match", "case", "break", "continue", "loop",
        "while", "for", "if", "else", "elif", "try", "catch", "throw",
        "import", "export", "from", "require", "module", "use", "mod",
        "self", "super", "true", "false", "null", "none", "nil", "void",
        "int", "str", "bool", "float", "vec", "map", "set", "list", "dict",
        "assert", "assert_eq", "println", "print", "printf", "console", "log",
    ].into_iter().collect();
    code_keywords.contains(term)
}

/// Extract key terms from text, filtering stopwords and code patterns
fn extract_key_terms(text: &str, max_terms: usize) -> Vec<String> {
    let stopwords: std::collections::HashSet<&str> = [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
        "because", "until", "while", "this", "that", "these", "those", "it",
        "its", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "any", "both",
        "also", "about", "like", "using", "based", "within", "without",
    ].into_iter().collect();

    let mut term_counts: HashMap<String, usize> = HashMap::new();

    for word in text.split(|c: char| !c.is_alphanumeric()) {
        let lower = word.to_lowercase();
        // Skip short words, stopwords, pure numbers, and code-like patterns
        if lower.len() >= 4
            && !stopwords.contains(lower.as_str())
            && !lower.chars().all(|c| c.is_numeric())
            && !is_code_like(&lower)
        {
            *term_counts.entry(lower).or_insert(0) += 1;
        }
    }

    // Sort by frequency and take top terms
    let mut terms: Vec<_> = term_counts.into_iter().collect();
    terms.sort_by(|a, b| b.1.cmp(&a.1));
    terms.into_iter().take(max_terms).map(|(t, _)| t).collect()
}

/// Expand query using terms from related passages
///
/// Takes the original query and texts from BM25-matched passages,
/// extracts key terms (including code symbols) and combines them with the original query.
pub fn expand_from_passages(query: &str, passage_texts: &[&str], max_expansion_terms: usize) -> String {
    if passage_texts.is_empty() {
        return query.to_string();
    }

    // Combine passage texts
    let combined_text = passage_texts.join(" ");

    // Extract key terms from prose
    let mut key_terms = extract_key_terms(&combined_text, max_expansion_terms);

    // Also extract code symbols (function/class names)
    let code_symbols = extract_code_symbols(&combined_text, max_expansion_terms);
    for symbol in code_symbols {
        if !key_terms.contains(&symbol.to_lowercase()) {
            key_terms.push(symbol);
        }
    }

    // Filter terms already in query
    let query_lower = query.to_lowercase();
    let query_words: std::collections::HashSet<_> = query_lower
        .split_whitespace()
        .collect();

    let new_terms: Vec<_> = key_terms
        .into_iter()
        .filter(|t| !query_words.contains(t.to_lowercase().as_str()))
        .take(max_expansion_terms)
        .collect();

    if new_terms.is_empty() {
        query.to_string()
    } else {
        format!("{} {}", query, new_terms.join(" "))
    }
}

/// Check if query expansion would be beneficial
/// Short queries (1-3 words) benefit most from expansion
pub fn should_expand(query: &str) -> bool {
    let word_count = query.split_whitespace().count();
    word_count <= 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_key_terms() {
        // Use repeated words to ensure they appear in top terms
        let text = "The architecture stores knowledge graph data. Architecture architecture knowledge knowledge.";
        let terms = extract_key_terms(text, 10);
        assert!(terms.contains(&"architecture".to_string()));
        assert!(terms.contains(&"knowledge".to_string()));
        assert!(!terms.contains(&"the".to_string())); // stopword
    }

    #[test]
    fn test_extract_key_terms_filters_code() {
        let text = "let graph = assert_eq edges test_case";
        let terms = extract_key_terms(text, 10);
        assert!(!terms.contains(&"let".to_string())); // code keyword
        assert!(!terms.contains(&"assert_eq".to_string())); // code keyword
        assert!(!terms.contains(&"test_case".to_string())); // underscore
        assert!(terms.contains(&"graph".to_string()) || terms.contains(&"edges".to_string()));
    }

    #[test]
    fn test_extract_code_symbols() {
        let text = r#"
            pub fn search_index(query: &str) -> Vec<Result> {}
            pub struct IndexSearcher { data: Vec<u8> }
            impl IndexSearcher {
                pub async fn load(&self) {}
            }
        "#;
        let symbols = extract_code_symbols(text, 10);
        assert!(symbols.contains(&"search_index".to_string()));
        assert!(symbols.contains(&"IndexSearcher".to_string()));
        assert!(symbols.contains(&"load".to_string()));
    }

    #[test]
    fn test_expand_from_passages() {
        let query = "database";
        let passages = vec![
            "Knowledge graph storage systems architecture",
            "Graph database for decisions planning",
        ];
        let expanded = expand_from_passages(query, &passages, 3);
        assert!(expanded.contains("database"));
        // Should contain at least one expanded term
        assert!(expanded.contains("knowledge") || expanded.contains("graph") ||
                expanded.contains("architecture") || expanded.contains("decisions"));
    }

    #[test]
    fn test_should_expand() {
        assert!(should_expand("database"));
        assert!(should_expand("graph db"));
        assert!(should_expand("api endpoint"));
        assert!(!should_expand("How to implement authentication in the API"));
    }
}
