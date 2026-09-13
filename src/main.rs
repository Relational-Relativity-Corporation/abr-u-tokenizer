// abr-u-tokenizer  main.rs
// CLI entry point for vocabulary build and smoke test.
//
// Usage:
//   abr_u_tokenizer build-vocab <u_units_index.txt> <output_dir>
//   abr_u_tokenizer tokenize   <u_units_index.txt> <p1.tsv> <p2.tsv> <text>
//   abr_u_tokenizer stats      <u_units_index.txt>

use abr_u_tokenizer::{
    load_participation_table, UTokenizer, UVocabulary,
};
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  {} build-vocab <u_units_index.txt> <output_dir>", args[0]);
        eprintln!("  {} tokenize <u_units_index.txt> <p1.tsv> <p2.tsv> <text>", args[0]);
        eprintln!("  {} stats <u_units_index.txt>", args[0]);
        std::process::exit(1);
    }

    match args[1].as_str() {
        "build-vocab" => cmd_build_vocab(&args),
        "tokenize"    => cmd_tokenize(&args),
        "stats"       => cmd_stats(&args),
        other => {
            eprintln!("Unknown command: {}", other);
            std::process::exit(1);
        }
    }
}

fn cmd_stats(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: {} stats <u_units_index.txt>", args[0]);
        std::process::exit(1);
    }
    let index_path = &args[2];
    let contents = fs::read_to_string(index_path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", index_path, e));
    let vocab = UVocabulary::from_index_file(&contents)
        .unwrap_or_else(|e| panic!("Vocab error: {}", e));

    println!("U unit vocabulary statistics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Total U units:      {}", vocab.u_unit_count);
    println!("Declared domain:    D_U = {}", vocab.u_unit_count);
    println!("OOV base ID:        {}", vocab.u_unit_count + 1);
    println!("First 10 units (sorted):");
    let mut units: Vec<(&String, &u32)> = vocab.unit_to_id.iter().collect();
    units.sort_by_key(|(_, &id)| id);
    for (unit, id) in units.iter().take(10) {
        println!("  {:4}  {:?}", id, unit);
    }
}

fn cmd_build_vocab(args: &[String]) {
    if args.len() < 4 {
        eprintln!("Usage: {} build-vocab <u_units_index.txt> <output_dir>", args[0]);
        std::process::exit(1);
    }
    let index_path = &args[2];
    let output_dir = &args[3];

    let contents = fs::read_to_string(index_path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", index_path, e));
    let vocab = UVocabulary::from_index_file(&contents)
        .unwrap_or_else(|e| panic!("Vocab error: {}", e));

    fs::create_dir_all(output_dir)
        .unwrap_or_else(|e| panic!("Cannot create output dir: {}", e));

    // Write vocab.txt: one unit per line, sorted by ID
    let mut units: Vec<(&String, &u32)> = vocab.unit_to_id.iter().collect();
    units.sort_by_key(|(_, &id)| id);
    let vocab_lines: Vec<String> = units
        .iter()
        .map(|(unit, id)| format!("{}\t{}", id, unit))
        .collect();
    let vocab_path = format!("{}/vocab.txt", output_dir);
    fs::write(&vocab_path, vocab_lines.join("\n") + "\n")
        .unwrap_or_else(|e| panic!("Cannot write vocab.txt: {}", e));

    // Write tokenizer_config.json
    let config = serde_json::json!({
        "tokenizer_class": "UTokenizer",
        "model_type": "abr-u-tokenizer",
        "version": "0.1.0",
        "vocab_size": vocab.u_unit_count,
        "oov_handling": "option_a_character_fallback",
        "oov_base_id": vocab.u_unit_count + 1,
        "framework": "ABR/ABRCE V7",
        "provenance": "D_U derived from strict local minima of Q(S) across 317,070 bounded structures",
        "declared_count": 5844
    });
    let config_path = format!("{}/tokenizer_config.json", output_dir);
    fs::write(
        &config_path,
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap_or_else(|e| panic!("Cannot write tokenizer_config.json: {}", e));

    println!("Vocabulary built successfully.");
    println!("  U units:       {}", vocab.u_unit_count);
    println!("  vocab.txt:     {}", vocab_path);
    println!("  config:        {}", config_path);
}

fn cmd_tokenize(args: &[String]) {
    if args.len() < 6 {
        eprintln!(
            "Usage: {} tokenize <u_units_index.txt> <p1.tsv> <p2.tsv> <text>",
            args[0]
        );
        std::process::exit(1);
    }
    let index_path = &args[2];
    let p1_path    = &args[3];
    let p2_path    = &args[4];
    let text       = &args[5];

    let index_contents = fs::read_to_string(index_path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", index_path, e));
    let p1_contents = fs::read_to_string(p1_path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", p1_path, e));
    let p2_contents = fs::read_to_string(p2_path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", p2_path, e));

    let vocab = UVocabulary::from_index_file(&index_contents)
        .unwrap_or_else(|e| panic!("Vocab error: {}", e));
    let p1 = load_participation_table(&p1_contents);
    let p2 = load_participation_table(&p2_contents);

    let tok = UTokenizer::new(vocab, p1, p2);

    println!("Input:   {:?}", text);
    let segments = tok.tokenize_to_strings(text);
    println!("Segments: {:?}", segments);
    let ids = tok.encode(text);
    println!("IDs:     {:?}", ids);
    let decoded = tok.decode(&ids);
    println!("Decoded: {:?}", decoded);
}
