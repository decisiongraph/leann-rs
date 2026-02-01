//! Integration tests for LEANN CLI

use std::process::Command;

fn cargo_run(args: &[&str]) -> std::process::Output {
    Command::new("cargo")
        .args(["run", "--quiet", "--"])
        .args(args)
        .output()
        .expect("Failed to run command")
}

#[test]
fn test_cli_help() {
    let output = cargo_run(&["--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("build"));
    assert!(stdout.contains("search"));
    assert!(stdout.contains("ask"));
    assert!(stdout.contains("react"));
    assert!(stdout.contains("serve"));
    assert!(stdout.contains("list"));
    assert!(stdout.contains("remove"));
}

#[test]
fn test_cli_version() {
    let output = cargo_run(&["--version"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("leann"));
}

#[test]
fn test_build_help() {
    let output = cargo_run(&["build", "--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--docs"));
    assert!(stdout.contains("--embedding-mode"));
    assert!(stdout.contains("--backend-name"));
}

#[test]
fn test_search_help() {
    let output = cargo_run(&["search", "--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--top-k"));
    assert!(stdout.contains("--filter"));
    assert!(stdout.contains("--hybrid"));
}

#[test]
fn test_ask_help() {
    let output = cargo_run(&["ask", "--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--llm"));
    assert!(stdout.contains("--model"));
    assert!(stdout.contains("--interactive"));
}

#[test]
fn test_react_help() {
    let output = cargo_run(&["react", "--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--max-steps"));
    assert!(stdout.contains("--verbose"));
}

#[test]
fn test_serve_help() {
    let output = cargo_run(&["serve", "--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--port"));
    assert!(stdout.contains("--cors"));
}
