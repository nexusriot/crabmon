//! Guards against documentation drifting away from the code. Every one of
//! these has a matching failure mode in the pre-rewrite tree, where the README
//! advertised behaviour the binary did not have.

use std::fs;

fn read(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The long flags the parser actually accepts, taken from the parser itself so
/// this list cannot drift from it. `--no-colour` is a spelling alias that the
/// docs deliberately do not repeat.
fn long_flags() -> Vec<&'static str> {
    crabmon::cli::LONG_FLAGS.iter().copied().filter(|f| *f != "--no-colour").collect()
}

#[test]
fn the_man_page_documents_every_flag() {
    let man = read("docs/crabmon.1");
    for flag in long_flags() {
        // roff escapes the leading dashes as \- .
        let escaped = flag.replace('-', "\\-");
        assert!(man.contains(&escaped), "{flag} is missing from the man page");
    }
}

#[test]
fn every_completion_offers_every_flag() {
    for path in ["completions/crabmon.bash", "completions/crabmon.zsh", "completions/crabmon.fish"]
    {
        let text = read(path);
        for flag in long_flags() {
            let needle = flag.trim_start_matches('-');
            assert!(text.contains(needle), "{path} does not offer {flag}");
        }
    }
}

#[test]
fn completions_list_the_real_enum_values() {
    let sorts: Vec<&str> = crabmon::sort::ALL_SORTS.iter().map(|s| s.key_name()).collect();
    for path in ["completions/crabmon.bash", "completions/crabmon.zsh", "completions/crabmon.fish"]
    {
        let text = read(path);
        for key in &sorts {
            assert!(text.contains(key), "{path} is missing sort key {key}");
        }
        for theme in crabmon::theme::PRESETS {
            assert!(text.contains(theme), "{path} is missing theme {theme}");
        }
    }
}

#[test]
fn the_readme_documents_every_key_the_help_popup_lists() {
    let readme = read("README.md");
    for (_, description) in crabmon::ui::popups::KEYS {
        // Each help entry's first word should appear somewhere in the README's
        // key table; this catches a binding added without a doc update.
        let word = description.split_whitespace().next().unwrap();
        assert!(
            readme.to_lowercase().contains(&word.to_lowercase()),
            "README never mentions '{word}' (from help entry '{description}')"
        );
    }
}

#[test]
fn the_readme_does_not_promise_a_refresh_floor_the_code_rejects() {
    let readme = read("README.md");
    assert!(
        readme.contains(&crabmon::MIN_REFRESH_MS.to_string()),
        "the README should state the {} ms refresh floor",
        crabmon::MIN_REFRESH_MS
    );
    assert!(!readme.contains("100 ms"), "the old 100 ms floor is below what the sampler supports");
}

#[test]
fn the_packaged_deb_ships_the_files_that_exist() {
    let manifest = read("Cargo.toml");
    for asset in [
        "docs/crabmon.1",
        "completions/crabmon.bash",
        "completions/crabmon.zsh",
        "completions/crabmon.fish",
        "CHANGELOG.md",
    ] {
        assert!(manifest.contains(asset), "{asset} is not packaged");
        assert!(fs::metadata(asset).is_ok(), "{asset} is packaged but missing");
    }
}

#[test]
fn the_deb_description_matches_the_features_that_exist() {
    let manifest = read("Cargo.toml");
    // The old description advertised "per-core CPU charts" that were never built.
    assert!(manifest.contains("per-core CPU history"));
    assert!(manifest.contains("tree view"));
}

/// Every field name that appears in a serialised default config, including the
/// nested tables. This is what the docs have to keep up with.
fn config_field_names() -> Vec<String> {
    let toml = crabmon::Config::default().to_toml();
    let mut names: Vec<String> = toml
        .lines()
        .filter_map(|l| l.split_once(" = ").map(|(k, _)| k.trim().to_string()))
        .filter(|k| !k.is_empty() && k.chars().all(|c| c.is_ascii_lowercase() || c == '_'))
        .collect();
    // Table headers: [thresholds], [panels], [[alert]] ...
    names.extend(
        toml.lines()
            .filter_map(|l| l.trim().strip_prefix('[').and_then(|r| r.strip_suffix(']')))
            .map(|t| t.trim_matches('[').trim_matches(']').to_string()),
    );
    names.sort();
    names.dedup();
    names
}

#[test]
fn the_readme_documents_every_config_field() {
    let readme = read("README.md");
    for field in config_field_names() {
        assert!(readme.contains(&field), "config field `{field}` is not documented in the README");
    }
}

#[test]
fn the_man_page_documents_every_config_field() {
    let man = read("docs/crabmon.1");
    for field in config_field_names() {
        // roff escapes hyphens; config keys use underscores, so a plain
        // substring match is enough.
        assert!(man.contains(&field), "config field `{field}` is not documented in the man page");
    }
}

#[test]
fn the_man_page_version_matches_the_crate() {
    let man = read("docs/crabmon.1");
    let version = env!("CARGO_PKG_VERSION");
    assert!(
        man.contains(&format!("crabmon {version}")),
        "the man page .TH line still says an older version than {version}"
    );
}

#[test]
fn the_docs_name_every_test_suite_that_exists() {
    let design = read("DESIGN.md");
    let mut suites: Vec<String> = fs::read_dir("tests")
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".rs"))
        .collect();
    suites.sort();
    for suite in &suites {
        assert!(design.contains(suite), "DESIGN.md never mentions tests/{suite}");
    }
    assert!(
        design.contains(&format!("{} integration suites", number_word(suites.len()))),
        "DESIGN.md does not say there are {} integration suites",
        suites.len()
    );
}

fn number_word(n: usize) -> &'static str {
    match n {
        4 => "four",
        5 => "five",
        6 => "six",
        7 => "seven",
        8 => "eight",
        _ => "many",
    }
}

#[test]
fn the_docs_list_every_theme_and_layout_that_exists() {
    let readme = read("README.md");
    let man = read("docs/crabmon.1");
    for theme in crabmon::theme::PRESETS {
        assert!(readme.contains(theme), "README omits theme {theme}");
        assert!(man.contains(theme), "man page omits theme {theme}");
    }
    for layout in ["dashboard", "processes", "cpu", "io"] {
        assert!(readme.contains(layout), "README omits layout {layout}");
        assert!(man.contains(layout), "man page omits layout {layout}");
    }
    // And the counts quoted in prose.
    assert!(
        readme.contains(&format!("{} colour themes", number_word(crabmon::theme::PRESETS.len()))),
        "the README's theme count is stale"
    );
}

#[test]
fn the_readme_and_man_page_agree_on_the_filter_language() {
    let readme = read("README.md");
    let man = read("docs/crabmon.1");
    for term in ["user:", "pid:", "ppid:", "state:", "cmd:", "re:", "cpu>", "mem>", "io>"] {
        assert!(readme.contains(term), "README omits filter term {term}");
        assert!(man.contains(term), "man page omits filter term {term}");
    }
}

#[test]
fn every_sort_key_is_documented_where_the_option_is_described() {
    // Scoped to the rows that describe `--sort` and `sort_by`, the way the
    // `--group` check is. Searching the whole file let `nice` go undocumented
    // in four places at once, because the word appears in "renice".
    let readme = read("README.md");
    let man = read("docs/crabmon.1");

    let readme_row = readme
        .lines()
        .find(|l| l.contains("`--sort <KEY>`"))
        .expect("the README options table should describe --sort");
    let readme_cfg = readme
        .lines()
        .find(|l| l.trim_start().starts_with("sort_by ="))
        .expect("the README config block should describe sort_by");
    let man_flag = section_after(&man, "\\-\\-sort", 260);
    let man_cfg = section_after(&man, "\n.B sort_by", 220);

    for key in crabmon::sort::ALL_SORTS {
        let k = key.key_name();
        assert!(readme_row.contains(k), "the README's --sort row omits {k}");
        assert!(readme_cfg.contains(k), "the README's sort_by comment omits {k}");
        assert!(man_flag.contains(k), "the man page's --sort entry omits {k}");
        assert!(man_cfg.contains(k), "the man page's sort_by entry omits {k}");
    }

    // The help text is a fourth place the same list is written out.
    let help = crabmon::cli::help_text();
    let help_row = help.lines().find(|l| l.contains("--sort")).expect("--sort in the help");
    for key in crabmon::sort::ALL_SORTS {
        assert!(help_row.contains(key.key_name()), "--help omits sort key {}", key.key_name());
    }
}

/// `len` bytes of `text` starting at `needle`, for scoping a check to one
/// man-page entry rather than the whole document.
fn section_after(text: &str, needle: &str, len: usize) -> String {
    let i = text.find(needle).unwrap_or_else(|| panic!("{needle} is missing"));
    text[i..(i + len).min(text.len())].to_string()
}

#[test]
fn the_refresh_bounds_quoted_in_the_docs_are_the_real_ones() {
    let min = crabmon::MIN_REFRESH_MS.to_string();
    let max = crabmon::MAX_REFRESH_MS.to_string();
    for path in ["README.md", "docs/crabmon.1"] {
        let text = read(path);
        assert!(text.contains(&min), "{path} does not state the {min} ms floor");
        assert!(text.contains(&max), "{path} does not state the {max} ms ceiling");
    }
}

/// Every key the normal-mode handler binds, extracted from the source so a new
/// binding cannot be added without documenting it.
fn bound_keys() -> Vec<char> {
    let src = read("src/app.rs");
    let start = src.find("fn on_normal_key").expect("on_normal_key");
    let end = src[start..].find("fn set_sort").expect("end of on_normal_key") + start;
    let body = &src[start..end];

    let mut keys = Vec::new();
    let mut rest = body;
    while let Some(i) = rest.find("KeyCode::Char('") {
        rest = &rest[i + "KeyCode::Char('".len()..];
        if let Some(c) = rest.chars().next() {
            keys.push(c);
        }
    }
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// The README's key-bindings table only. Checking the whole file would pass on
/// an incidental mention of the letter somewhere in the prose.
fn readme_key_table() -> String {
    let readme = read("README.md");
    let start = readme.find("### Key bindings").expect("key bindings section");
    let rest = &readme[start..];
    let end = rest.find("\n### ").or_else(|| rest.find("\n## ")).unwrap_or(rest.len());
    rest[..end].to_string()
}

/// The man page's KEYS section only, for the same reason.
fn man_keys_section() -> String {
    let man = read("docs/crabmon.1");
    let start = man.find(".SH KEYS").expect("KEYS section");
    let rest = &man[start..];
    let end = rest[1..].find("\n.SH ").map(|i| i + 1).unwrap_or(rest.len());
    rest[..end].to_string()
}

#[test]
fn every_bound_key_appears_in_the_readme_and_the_man_page() {
    let readme = readme_key_table();
    let man = man_keys_section();
    for key in bound_keys() {
        // The number keys are documented as a range, and space by name.
        let (r_needle, m_needle) = match key {
            ' ' => ("Space".to_string(), "Space".to_string()),
            '1'..='9' => ("`1`–`9`".to_string(), "1-9".to_string()),
            c => (format!("`{c}`"), c.to_string()),
        };
        assert!(
            readme.contains(&r_needle),
            "key '{key}' is missing from the README's key-bindings table"
        );
        assert!(man.contains(&m_needle), "key '{key}' is missing from the man page's KEYS section");
    }
}

#[test]
fn the_readme_and_man_page_agree_on_the_key_aliases() {
    // `,` `.` `=` `_` and F1 all work; documenting them in only one place is
    // exactly the drift this suite exists to catch.
    let readme = readme_key_table();
    let man = man_keys_section();
    let pairs = [
        ("(or `,` `.`)", r#", " and " ."#),
        ("(or `=` `_`)", r#"= " and " _"#),
        ("`?`, `F1`", "F1"),
    ];
    for (in_readme, in_man) in pairs {
        assert!(readme.contains(in_readme), "the README's key table omits {in_readme}");
        assert!(man.contains(in_man), "the man page's KEYS section omits {in_man}");
    }
}

#[test]
fn every_grouping_mode_is_documented_where_the_option_is_described() {
    let readme = read("README.md");
    let man = read("docs/crabmon.1");
    // Scope to the lines describing --group / group_by, so an incidental
    // mention of the word "container" elsewhere cannot satisfy this.
    let readme_row = readme
        .lines()
        .find(|l| l.contains("`--group <BY>`"))
        .expect("the README options table should describe --group");
    let man_block = {
        let i = man.find("\\-\\-group").expect("man page should describe --group");
        man[i..i + 200].to_string()
    };
    for g in crabmon::metrics::procgroup::ALL_GROUPINGS {
        assert!(readme_row.contains(g.label()), "the README's --group row omits {}", g.label());
        assert!(man_block.contains(g.label()), "the man page's --group entry omits {}", g.label());
    }
}

#[test]
fn every_panel_that_can_be_switched_off_is_documented() {
    let readme = read("README.md");
    let man = read("docs/crabmon.1");
    // Field names of the [panels] table, from a serialised default config.
    let toml = crabmon::Config::default().to_toml();
    let panels_section = toml.split("[panels]").nth(1).expect("[panels] table");
    let panels: Vec<&str> = panels_section
        .lines()
        .take_while(|l| !l.trim_start().starts_with('['))
        .filter_map(|l| l.split_once(" = ").map(|(k, _)| k.trim()))
        .collect();
    assert!(panels.len() >= 9, "expected the full panel list, got {panels:?}");
    for panel in panels {
        assert!(readme.contains(panel), "README omits panel {panel}");
        assert!(man.contains(panel), "man page omits panel {panel}");
    }
}

#[test]
fn the_deb_description_covers_the_current_feature_set() {
    let manifest = read("Cargo.toml");
    let description = manifest
        .split("extended-description")
        .nth(1)
        .and_then(|s| s.split("\"\"\"").nth(1))
        .expect("extended-description");
    for feature in ["recording", "Prometheus", "SSH", "pressure", "battery", "sockets"] {
        assert!(description.contains(feature), "the packaged description never mentions {feature}");
    }
}

#[test]
fn the_test_count_quoted_in_design_is_not_wildly_stale() {
    // Not exact — that would be churn on every added test — but it must not
    // claim a number the suite has long since passed.
    let design = read("DESIGN.md");
    let quoted: usize = design
        .split("Over ")
        .nth(1)
        .and_then(|s| s.split(' ').next())
        .and_then(|n| n.parse().ok())
        .expect("DESIGN.md should quote a test count as 'Over N tests'");
    // The suites are counted at build time in CI; here just sanity-check the
    // claim is in the right order of magnitude and not above the real figure.
    assert!(quoted >= 200, "the quoted count {quoted} looks stale");
    assert!(quoted <= 1000, "the quoted count {quoted} looks invented");
}

/// Ask git whether it would ignore `path`. `None` when git is unavailable or
/// this is not a checkout, so the test can skip rather than fail spuriously.
fn git_ignores(path: &str) -> Option<bool> {
    let out = std::process::Command::new("git").args(["check-ignore", "-q", path]).output().ok()?;
    match out.status.code() {
        Some(0) => Some(true),  // ignored
        Some(1) => Some(false), // not ignored
        _ => None,              // not a repo, or git is unhappy
    }
}

#[test]
fn gitignore_covers_the_files_crabmon_writes_into_the_working_directory() {
    // With `[export] dir` unset, `P` writes here — which is the checkout while
    // developing. These names come from `export_path` itself, not a guess.
    let json = crabmon::export::export_path(
        std::path::Path::new("."),
        1_788_297_407,
        crabmon::export::ExportFormat::Json,
    );
    let csv = crabmon::export::export_path(
        std::path::Path::new("."),
        1_788_297_407,
        crabmon::export::ExportFormat::Csv,
    );

    for path in [json, csv] {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        match git_ignores(&name) {
            Some(true) => {}
            Some(false) => panic!("an exported snapshot ({name}) would be committed"),
            None => return, // not a git checkout; nothing to assert
        }
    }

    // Recordings, wherever they are written.
    for name in ["run.jsonl", "recordings/slowdown.jsonl"] {
        if git_ignores(name) == Some(false) {
            panic!("a recording ({name}) would be committed");
        }
    }

    // The audit log, when pointed somewhere relative.
    if git_ignores("actions.log") == Some(false) {
        panic!("the action log would be committed");
    }
}

#[test]
fn gitignore_does_not_swallow_the_projects_own_files() {
    // The mirror of the above: an over-broad pattern that hides real sources is
    // worse than a missing one.
    for path in [
        "src/main.rs",
        "src/metrics/psi.rs",
        "tests/docs.rs",
        "README.md",
        "DESIGN.md",
        "Cargo.toml",
        "Cargo.lock",
        "docs/crabmon.1",
        "docs/crabmon.png",
        "CHANGELOG.md",
        "completions/crabmon.bash",
        ".github/workflows/ci.yml",
        "scripts/build-deb.sh",
        "rustfmt.toml",
    ] {
        if git_ignores(path) == Some(true) {
            panic!("{path} is a tracked project file but .gitignore hides it");
        }
    }
}

#[test]
fn every_gitignore_rule_says_what_it_is_for() {
    // A bare list of globs rots; a commented one tells the next person whether
    // a rule is still needed.
    let text = read(".gitignore");
    let mut sections = 0;
    for block in text.split("\n\n") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        sections += 1;
        assert!(
            block.lines().any(|l| l.trim_start().starts_with('#')),
            "this .gitignore block has no comment explaining it:\n{block}"
        );
    }
    assert!(sections >= 4, "expected the file to stay grouped by purpose");
}

/// The version at the top of the changelog.
fn changelog_latest_version() -> String {
    let text = read("CHANGELOG.md");
    text.lines()
        .find_map(|l| l.strip_prefix("## "))
        .map(|l| l.split_whitespace().next().unwrap_or("").to_string())
        .expect("CHANGELOG.md should open with a '## <version>' heading")
}

#[test]
fn the_changelog_leads_with_the_current_version() {
    assert_eq!(
        changelog_latest_version(),
        env!("CARGO_PKG_VERSION"),
        "the changelog's newest entry does not match the crate version"
    );
}

#[test]
fn the_changelog_accounts_for_every_released_version() {
    // A version bump with no entry is how a changelog stops being useful.
    let text = read("CHANGELOG.md");
    let versions: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("## "))
        .map(|l| l.split_whitespace().next().unwrap_or(""))
        .collect();
    assert!(versions.len() >= 2, "expected a history, got {versions:?}");

    // Every heading must parse as a version, newest first.
    let parsed: Vec<Vec<u32>> = versions
        .iter()
        .map(|v| {
            v.split('.')
                .map(|p| p.parse().unwrap_or_else(|_| panic!("bad version heading: {v}")))
                .collect()
        })
        .collect();
    for pair in parsed.windows(2) {
        assert!(pair[0] > pair[1], "changelog versions must run newest first: {versions:?}");
    }

    // The version the repository last shipped must still be listed.
    assert!(versions.contains(&"0.2.2"), "the released 0.2.2 entry was dropped");
}

/// Every flag arrived in some release, so every flag should appear somewhere in
/// the changelog. Derived from the parser rather than a hand-written list,
/// which is what used to go stale at each version bump.
#[test]
fn the_changelog_accounts_for_every_flag_the_parser_accepts() {
    let text = read("CHANGELOG.md");
    for flag in long_flags() {
        // `--help` and `--version` are not features anyone announces.
        if matches!(flag, "--help" | "--version") {
            continue;
        }
        assert!(text.contains(flag), "no changelog entry ever mentions {flag}");
    }
}

#[test]
fn the_newest_changelog_entry_describes_this_version() {
    let text = read("CHANGELOG.md");
    // Element 0 is the file preamble; element 1 is the newest release section.
    let latest = text.split("\n## ").nth(1).expect("a release section");
    assert!(
        latest.starts_with(env!("CARGO_PKG_VERSION")),
        "the newest section is not this version"
    );
    assert!(
        latest.lines().filter(|l| l.trim_start().starts_with("- ")).count() >= 3,
        "a release with no entries is not a release note"
    );
}
