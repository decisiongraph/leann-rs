//! Metadata filtering for search results

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Filter operator
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FilterOp {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
    NotIn,
    Contains,
    StartsWith,
    EndsWith,
    Exists,
}

/// A single filter condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    pub field: String,
    pub op: FilterOp,
    pub value: Value,
}

/// Combined filter with AND/OR logic
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetadataFilter {
    Condition(FilterCondition),
    And { and: Vec<MetadataFilter> },
    Or { or: Vec<MetadataFilter> },
}

impl MetadataFilter {
    /// Parse a filter string with support for:
    /// - Simple: "source:*.rs", "type=code", "lines>50"
    /// - Multiple (AND): "type=code,lines>50" or "type=code AND lines>50"
    /// - Multiple (OR): "type=code OR type=text"
    /// - In: "type in [code,text,doc]"
    /// - Not in: "type not_in [code,text]"
    /// - Contains: "source~keyword" or "source:*keyword*"
    /// - Starts with: "source^prefix" or "source:prefix*"
    /// - Ends with: "source$suffix" or "source:*suffix"
    /// - Exists: "field?"
    pub fn parse(filter_str: &str) -> Option<Self> {
        let filter_str = filter_str.trim();

        // Check for OR (lower precedence)
        if filter_str.contains(" OR ") {
            let parts: Vec<&str> = filter_str.split(" OR ").collect();
            let filters: Vec<MetadataFilter> = parts
                .iter()
                .filter_map(|p| MetadataFilter::parse(p.trim()))
                .collect();
            if filters.len() > 1 {
                return Some(MetadataFilter::Or { or: filters });
            } else if filters.len() == 1 {
                return Some(filters.into_iter().next().unwrap());
            }
            return None;
        }

        // Check for AND (comma or explicit AND)
        // But avoid splitting commas inside [...] brackets
        let has_and = filter_str.contains(" AND ");
        let has_comma_outside_brackets = {
            let mut depth = 0;
            let mut found = false;
            for c in filter_str.chars() {
                match c {
                    '[' => depth += 1,
                    ']' => depth -= 1,
                    ',' if depth == 0 => {
                        found = true;
                        break;
                    }
                    _ => {}
                }
            }
            found
        };

        if has_and || has_comma_outside_brackets {
            let parts: Vec<String> = if has_and {
                filter_str.split(" AND ").map(|s| s.to_string()).collect()
            } else {
                // Split only commas outside brackets
                let mut parts = Vec::new();
                let mut current = String::new();
                let mut depth = 0;
                for c in filter_str.chars() {
                    match c {
                        '[' => {
                            depth += 1;
                            current.push(c);
                        }
                        ']' => {
                            depth -= 1;
                            current.push(c);
                        }
                        ',' if depth == 0 => {
                            parts.push(current.clone());
                            current.clear();
                        }
                        _ => current.push(c),
                    }
                }
                if !current.is_empty() {
                    parts.push(current);
                }
                parts
            };
            let filters: Vec<MetadataFilter> = parts
                .iter()
                .filter_map(|p| MetadataFilter::parse_single(p.trim()))
                .collect();
            if filters.len() > 1 {
                return Some(MetadataFilter::And { and: filters });
            } else if filters.len() == 1 {
                return Some(filters.into_iter().next().unwrap());
            }
            return None;
        }

        // Single condition
        Self::parse_single(filter_str)
    }

    /// Parse a single filter condition
    fn parse_single(filter_str: &str) -> Option<Self> {
        let filter_str = filter_str.trim();

        // Check for "exists" operator: field?
        if filter_str.ends_with('?') {
            return Some(MetadataFilter::Condition(FilterCondition {
                field: filter_str[..filter_str.len() - 1].to_string(),
                op: FilterOp::Exists,
                value: Value::Null,
            }));
        }

        // Check for "in" operator: field in [a,b,c]
        if let Some(idx) = filter_str.find(" in [") {
            let field = filter_str[..idx].trim().to_string();
            let rest = &filter_str[idx + 5..];
            if let Some(end) = rest.find(']') {
                let values: Vec<Value> = rest[..end]
                    .split(',')
                    .map(|v| parse_value(v.trim()))
                    .collect();
                return Some(MetadataFilter::Condition(FilterCondition {
                    field,
                    op: FilterOp::In,
                    value: Value::Array(values),
                }));
            }
        }

        // Check for "not_in" operator: field not_in [a,b,c]
        if let Some(idx) = filter_str.find(" not_in [") {
            let field = filter_str[..idx].trim().to_string();
            let rest = &filter_str[idx + 9..];
            if let Some(end) = rest.find(']') {
                let values: Vec<Value> = rest[..end]
                    .split(',')
                    .map(|v| parse_value(v.trim()))
                    .collect();
                return Some(MetadataFilter::Condition(FilterCondition {
                    field,
                    op: FilterOp::NotIn,
                    value: Value::Array(values),
                }));
            }
        }

        // Check for contains operator: field~value
        if filter_str.contains('~') {
            let p: Vec<&str> = filter_str.splitn(2, '~').collect();
            if p.len() == 2 {
                return Some(MetadataFilter::Condition(FilterCondition {
                    field: p[0].to_string(),
                    op: FilterOp::Contains,
                    value: Value::String(p[1].to_string()),
                }));
            }
            return None;
        }

        // Check for starts_with operator: field^value
        if filter_str.contains('^') && !filter_str.contains(">=") {
            let p: Vec<&str> = filter_str.splitn(2, '^').collect();
            if p.len() == 2 {
                return Some(MetadataFilter::Condition(FilterCondition {
                    field: p[0].to_string(),
                    op: FilterOp::StartsWith,
                    value: Value::String(p[1].to_string()),
                }));
            }
            return None;
        }

        // Check for ends_with operator: field$value
        if filter_str.contains('$') {
            let p: Vec<&str> = filter_str.splitn(2, '$').collect();
            if p.len() == 2 {
                return Some(MetadataFilter::Condition(FilterCondition {
                    field: p[0].to_string(),
                    op: FilterOp::EndsWith,
                    value: Value::String(p[1].to_string()),
                }));
            }
            return None;
        }

        // Format: field:value or field=value or field>value etc.
        let parts: Vec<&str> = if filter_str.contains("!=") {
            let p: Vec<&str> = filter_str.splitn(2, "!=").collect();
            if p.len() == 2 {
                return Some(MetadataFilter::Condition(FilterCondition {
                    field: p[0].to_string(),
                    op: FilterOp::Ne,
                    value: parse_value(p[1]),
                }));
            }
            return None;
        } else if filter_str.contains(">=") {
            let p: Vec<&str> = filter_str.splitn(2, ">=").collect();
            if p.len() == 2 {
                return Some(MetadataFilter::Condition(FilterCondition {
                    field: p[0].to_string(),
                    op: FilterOp::Gte,
                    value: parse_value(p[1]),
                }));
            }
            return None;
        } else if filter_str.contains("<=") {
            let p: Vec<&str> = filter_str.splitn(2, "<=").collect();
            if p.len() == 2 {
                return Some(MetadataFilter::Condition(FilterCondition {
                    field: p[0].to_string(),
                    op: FilterOp::Lte,
                    value: parse_value(p[1]),
                }));
            }
            return None;
        } else if filter_str.contains('>') {
            let p: Vec<&str> = filter_str.splitn(2, '>').collect();
            if p.len() == 2 {
                return Some(MetadataFilter::Condition(FilterCondition {
                    field: p[0].to_string(),
                    op: FilterOp::Gt,
                    value: parse_value(p[1]),
                }));
            }
            return None;
        } else if filter_str.contains('<') {
            let p: Vec<&str> = filter_str.splitn(2, '<').collect();
            if p.len() == 2 {
                return Some(MetadataFilter::Condition(FilterCondition {
                    field: p[0].to_string(),
                    op: FilterOp::Lt,
                    value: parse_value(p[1]),
                }));
            }
            return None;
        } else if filter_str.contains('=') {
            filter_str.splitn(2, '=').collect()
        } else if filter_str.contains(':') {
            filter_str.splitn(2, ':').collect()
        } else {
            return None;
        };

        if parts.len() != 2 {
            return None;
        }

        let field = parts[0].to_string();
        let value = parts[1];

        // Check for glob patterns
        if value.contains('*') {
            if value.starts_with('*') && value.ends_with('*') && value.len() > 2 {
                return Some(MetadataFilter::Condition(FilterCondition {
                    field,
                    op: FilterOp::Contains,
                    value: Value::String(value[1..value.len() - 1].to_string()),
                }));
            } else if value.starts_with('*') {
                return Some(MetadataFilter::Condition(FilterCondition {
                    field,
                    op: FilterOp::EndsWith,
                    value: Value::String(value[1..].to_string()),
                }));
            } else if value.ends_with('*') {
                return Some(MetadataFilter::Condition(FilterCondition {
                    field,
                    op: FilterOp::StartsWith,
                    value: Value::String(value[..value.len() - 1].to_string()),
                }));
            }
        }

        Some(MetadataFilter::Condition(FilterCondition {
            field,
            op: FilterOp::Eq,
            value: parse_value(value),
        }))
    }

    /// Check if metadata matches this filter
    pub fn matches(&self, metadata: &Value) -> bool {
        match self {
            MetadataFilter::Condition(cond) => cond.matches(metadata),
            MetadataFilter::And { and } => and.iter().all(|f| f.matches(metadata)),
            MetadataFilter::Or { or } => or.iter().any(|f| f.matches(metadata)),
        }
    }
}

impl FilterCondition {
    fn matches(&self, metadata: &Value) -> bool {
        let field_value = get_nested_value(metadata, &self.field);

        match &self.op {
            FilterOp::Exists => field_value.is_some(),
            FilterOp::Eq => field_value.map_or(false, |v| values_equal(v, &self.value)),
            FilterOp::Ne => field_value.map_or(true, |v| !values_equal(v, &self.value)),
            FilterOp::Gt => field_value.map_or(false, |v| compare_values(v, &self.value) > 0),
            FilterOp::Gte => field_value.map_or(false, |v| compare_values(v, &self.value) >= 0),
            FilterOp::Lt => field_value.map_or(false, |v| compare_values(v, &self.value) < 0),
            FilterOp::Lte => field_value.map_or(false, |v| compare_values(v, &self.value) <= 0),
            FilterOp::In => {
                if let Some(arr) = self.value.as_array() {
                    field_value.map_or(false, |v| arr.iter().any(|item| values_equal(v, item)))
                } else {
                    false
                }
            }
            FilterOp::NotIn => {
                if let Some(arr) = self.value.as_array() {
                    field_value.map_or(true, |v| !arr.iter().any(|item| values_equal(v, item)))
                } else {
                    true
                }
            }
            FilterOp::Contains => {
                let pattern = self.value.as_str().unwrap_or("");
                field_value
                    .and_then(|v| v.as_str())
                    .map_or(false, |s| s.contains(pattern))
            }
            FilterOp::StartsWith => {
                let pattern = self.value.as_str().unwrap_or("");
                field_value
                    .and_then(|v| v.as_str())
                    .map_or(false, |s| s.starts_with(pattern))
            }
            FilterOp::EndsWith => {
                let pattern = self.value.as_str().unwrap_or("");
                field_value
                    .and_then(|v| v.as_str())
                    .map_or(false, |s| s.ends_with(pattern))
            }
        }
    }
}

fn get_nested_value<'a>(metadata: &'a Value, path: &str) -> Option<&'a Value> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = metadata;

    for part in parts {
        match current.get(part) {
            Some(v) => current = v,
            None => return None,
        }
    }

    Some(current)
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::String(s1), Value::String(s2)) => s1 == s2,
        (Value::Number(n1), Value::Number(n2)) => {
            n1.as_f64().zip(n2.as_f64()).map_or(false, |(a, b)| (a - b).abs() < f64::EPSILON)
        }
        (Value::Bool(b1), Value::Bool(b2)) => b1 == b2,
        (Value::Null, Value::Null) => true,
        _ => false,
    }
}

fn compare_values(a: &Value, b: &Value) -> i32 {
    match (a.as_f64(), b.as_f64()) {
        (Some(n1), Some(n2)) => {
            if n1 < n2 {
                -1
            } else if n1 > n2 {
                1
            } else {
                0
            }
        }
        _ => match (a.as_str(), b.as_str()) {
            (Some(s1), Some(s2)) => s1.cmp(s2) as i32,
            _ => 0,
        },
    }
}

fn parse_value(s: &str) -> Value {
    // Try to parse as number
    if let Ok(n) = s.parse::<i64>() {
        return Value::Number(n.into());
    }
    if let Ok(n) = s.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(n) {
            return Value::Number(n);
        }
    }
    // Try to parse as bool
    if s == "true" {
        return Value::Bool(true);
    }
    if s == "false" {
        return Value::Bool(false);
    }
    // Default to string
    Value::String(s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_filter_parse() {
        let filter = MetadataFilter::parse("source:*.rs").unwrap();
        assert!(matches!(filter, MetadataFilter::Condition(_)));
    }

    #[test]
    fn test_filter_matches() {
        let metadata = json!({
            "source": "main.rs",
            "type": "code",
            "lines": 100
        });

        let filter = MetadataFilter::parse("source:*.rs").unwrap();
        assert!(filter.matches(&metadata));

        let filter = MetadataFilter::parse("type=code").unwrap();
        assert!(filter.matches(&metadata));

        let filter = MetadataFilter::parse("lines>50").unwrap();
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_filter_in() {
        let metadata = json!({
            "type": "code",
            "lang": "rust"
        });

        let filter = MetadataFilter::parse("type in [code,text,doc]").unwrap();
        assert!(filter.matches(&metadata));

        let filter = MetadataFilter::parse("type in [text,doc]").unwrap();
        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_filter_not_in() {
        let metadata = json!({
            "type": "code"
        });

        let filter = MetadataFilter::parse("type not_in [text,doc]").unwrap();
        assert!(filter.matches(&metadata));

        let filter = MetadataFilter::parse("type not_in [code,text]").unwrap();
        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_filter_and() {
        let metadata = json!({
            "type": "code",
            "lines": 100
        });

        let filter = MetadataFilter::parse("type=code,lines>50").unwrap();
        assert!(filter.matches(&metadata));

        let filter = MetadataFilter::parse("type=code AND lines>50").unwrap();
        assert!(filter.matches(&metadata));

        let filter = MetadataFilter::parse("type=code,lines>200").unwrap();
        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_filter_or() {
        let metadata = json!({
            "type": "code"
        });

        let filter = MetadataFilter::parse("type=code OR type=text").unwrap();
        assert!(filter.matches(&metadata));

        let filter = MetadataFilter::parse("type=text OR type=doc").unwrap();
        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_filter_contains() {
        let metadata = json!({
            "source": "/path/to/main.rs"
        });

        let filter = MetadataFilter::parse("source~main").unwrap();
        assert!(filter.matches(&metadata));

        let filter = MetadataFilter::parse("source:*main*").unwrap();
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_filter_exists() {
        let metadata = json!({
            "source": "main.rs"
        });

        let filter = MetadataFilter::parse("source?").unwrap();
        assert!(filter.matches(&metadata));

        let filter = MetadataFilter::parse("missing?").unwrap();
        assert!(!filter.matches(&metadata));
    }
}
