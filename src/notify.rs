//! Built-in alert notifications: `[[notify]]`.
//!
//! A hook can page anybody, but every user ends up writing the same `curl`
//! one-liner to post to Slack or ntfy, getting the JSON quoting wrong, and
//! putting a webhook URL — which is a credential — on a command line that
//! `ps` shows to every user on the machine. This is that one-liner, done once.
//!
//! The request is still made by `curl`, as a hook would make it: crabmon
//! links no HTTP or TLS stack, and `curl` is on every machine that has a
//! network worth posting from. What is new is that the URL, the token and
//! the body reach it on standard input as a `curl --config` file, never in
//! argv, and that it runs under the same reaping and timeout as any hook.

use serde::{Deserialize, Serialize};

use crate::alerts::HookCall;

/// One destination for alert transitions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NotifyTarget {
    /// What this target is called in notes and `--doctor`.
    pub name: String,
    /// `webhook`, `slack`, `discord` or `ntfy`.
    pub kind: String,
    pub url: String,
    /// Sent as `Authorization: Bearer <token>` when set: an ntfy access
    /// token, or whatever a generic webhook expects.
    pub token: String,
    /// Only these rules, by name. Empty means every rule.
    pub rules: Vec<String>,
    /// Post recoveries as well as firings.
    pub cleared: bool,
}

impl Default for NotifyTarget {
    fn default() -> Self {
        NotifyTarget {
            name: String::new(),
            kind: "webhook".into(),
            url: String::new(),
            token: String::new(),
            rules: Vec::new(),
            cleared: true,
        }
    }
}

pub const KINDS: [&str; 4] = ["webhook", "slack", "discord", "ntfy"];

/// How long `curl` may take over one post. Under the hook timeout, so a
/// slow endpoint is reported as a failed delivery rather than a killed hook.
pub const POST_TIMEOUT_SECS: u64 = 15;

impl NotifyTarget {
    /// The name notes use: the configured one, else the kind.
    pub fn label(&self) -> &str {
        if self.name.is_empty() { &self.kind } else { &self.name }
    }

    /// Whether this target wants to hear about `call`.
    pub fn wants(&self, call: &HookCall) -> bool {
        (call.firing || self.cleared) && (self.rules.is_empty() || self.rules.contains(&call.rule))
    }
}

/// Refuse a target that could never deliver anything, so it fails at start
/// rather than silently at three in the morning.
pub fn validate(targets: &[NotifyTarget]) -> Result<(), String> {
    for (i, t) in targets.iter().enumerate() {
        let which = match t.name.is_empty() {
            true => format!("[[notify]] #{}", i + 1),
            false => format!("[[notify]] {}", t.name),
        };
        if !KINDS.contains(&t.kind.as_str()) {
            return Err(format!("{which}: unknown kind {:?}; known: {}", t.kind, KINDS.join(", ")));
        }
        if !(t.url.starts_with("https://") || t.url.starts_with("http://")) {
            return Err(format!("{which}: url must start with https:// or http://"));
        }
        if t.url.chars().any(|c| c.is_control() || c.is_whitespace()) {
            return Err(format!("{which}: url contains whitespace or control characters"));
        }
    }
    Ok(())
}

/// A string as `curl --config` quotes one.
fn quoted(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A header value cannot carry a line break: one would end the header and
/// start another, chosen by whoever named the rule.
fn header_safe(s: &str) -> String {
    s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect()
}

/// The line a chat message leads with. Under a fleet the message already
/// names the host; on one machine it does not, and a page that does not say
/// which machine has to be answered twice.
fn headline(call: &HookCall) -> String {
    let state = if call.firing { "FIRING" } else { "RESOLVED" };
    match call.host.is_empty() || call.message.starts_with(&format!("{}:", call.host)) {
        true => format!("[{state}] {}", call.message),
        false => format!("[{state}] {}: {}", call.host, call.message),
    }
}

/// A chat message with the rule's evidence — a failing unit's last journal
/// lines — under it as a code block, so the page says why and not only what.
/// Discord refuses a message over 2000 characters, so the block is kept
/// well inside that, and a fence inside the log cannot close it early.
fn with_detail(text: String, call: &HookCall) -> String {
    if call.detail.is_empty() {
        return text;
    }
    let detail = crate::format::truncate_fit(&call.detail.replace("```", "'''"), 1500);
    format!("{text}\n```\n{detail}\n```")
}

/// The JSON a generic webhook receives: the hook's variables, lower-cased
/// and without the prefix, with the two numbers as numbers.
pub fn webhook_body(call: &HookCall) -> String {
    let mut map = serde_json::Map::new();
    for (name, value) in call.env() {
        let key = name.trim_start_matches("CRABMON_").to_ascii_lowercase();
        let value = match key.as_str() {
            "at" => value
                .parse::<u64>()
                .map(|n| serde_json::Value::Number(n.into()))
                .unwrap_or(serde_json::Value::String(value)),
            "value" | "threshold" => value
                .parse::<f64>()
                .ok()
                .and_then(serde_json::Number::from_f64)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::String(value)),
            _ => serde_json::Value::String(value),
        };
        map.insert(key, value);
    }
    serde_json::Value::Object(map).to_string()
}

/// The `curl --config` text that posts `call` to `target`.
pub fn curl_config(target: &NotifyTarget, call: &HookCall) -> String {
    let mut headers: Vec<String> = Vec::new();
    let body = match target.kind.as_str() {
        "slack" => {
            headers.push("Content-Type: application/json".into());
            serde_json::json!({ "text": with_detail(headline(call), call) }).to_string()
        }
        "discord" => {
            headers.push("Content-Type: application/json".into());
            serde_json::json!({ "content": with_detail(headline(call), call) }).to_string()
        }
        "ntfy" => {
            let host =
                if call.host.is_empty() { String::new() } else { format!(" on {}", call.host) };
            let state = if call.firing { "firing" } else { "cleared" };
            headers.push(format!("Title: crabmon: {} {state}{host}", call.rule));
            headers.push(format!("Priority: {}", if call.firing { "high" } else { "default" }));
            headers.push(format!(
                "Tags: {}",
                if call.firing { "warning" } else { "white_check_mark" }
            ));
            match call.detail.is_empty() {
                true => call.message.clone(),
                false => format!("{}\n\n{}", call.message, call.detail),
            }
        }
        _ => {
            headers.push("Content-Type: application/json".into());
            webhook_body(call)
        }
    };
    if !target.token.is_empty() {
        headers.push(format!("Authorization: Bearer {}", target.token));
    }
    let mut out = String::new();
    out.push_str(&format!("url = {}\n", quoted(&target.url)));
    out.push_str("request = \"POST\"\n");
    for h in headers {
        out.push_str(&format!("header = {}\n", quoted(&header_safe(&h))));
    }
    out.push_str(&format!("data-binary = {}\n", quoted(&body)));
    out.push_str(&format!("max-time = {POST_TIMEOUT_SECS}\n"));
    out.push_str("silent\nshow-error\nfail\noutput = \"/dev/null\"\n");
    out
}

/// Post every transition each target wants. Returns notes for anything that
/// could not be started; a failed delivery is reported when it is reaped.
pub fn send(targets: &[NotifyTarget], calls: &[HookCall]) -> Vec<String> {
    let mut notes = Vec::new();
    for target in targets {
        for call in calls.iter().filter(|c| target.wants(c)) {
            let label = format!("notify {}", target.label());
            notes.extend(crate::supervisor::spawn_curl(&label, curl_config(target, call)));
        }
    }
    notes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alerts::AlertKind;

    fn call(firing: bool) -> HookCall {
        HookCall {
            command: String::new(),
            rule: "cpu-saturated".into(),
            kind: AlertKind::Cpu,
            value: 93.5,
            threshold: 90.0,
            firing,
            message: "cpu-saturated 93.5% ≥ 90%".into(),
            at_unix: 1_700_000_000,
            host: "web-1".into(),
            dump: String::new(),
            summary: String::new(),
            detail: String::new(),
        }
    }

    fn target(kind: &str) -> NotifyTarget {
        NotifyTarget {
            kind: kind.into(),
            url: "https://example.test/hook".into(),
            ..Default::default()
        }
    }

    #[test]
    fn a_target_hears_only_the_rules_and_states_it_asked_for() {
        let mut t = target("slack");
        assert!(t.wants(&call(true)) && t.wants(&call(false)));
        t.cleared = false;
        assert!(t.wants(&call(true)) && !t.wants(&call(false)));
        t.rules = vec!["disk-full".into()];
        assert!(!t.wants(&call(true)));
    }

    #[test]
    fn targets_that_cannot_deliver_are_refused() {
        assert!(validate(&[target("slack")]).is_ok());
        assert!(validate(&[target("pager")]).unwrap_err().contains("unknown kind"));
        let mut t = target("ntfy");
        t.url = "ntfy.sh/alerts".into();
        assert!(validate(&[t.clone()]).unwrap_err().contains("https://"));
        t.url = "https://ntfy.sh/a b".into();
        t.name = "phone".into();
        assert!(validate(&[t]).unwrap_err().contains("[[notify]] phone"));
    }

    #[test]
    fn the_generic_webhook_gets_the_hook_variables_as_json() {
        let body: serde_json::Value = serde_json::from_str(&webhook_body(&call(true))).unwrap();
        assert_eq!(body["rule"], "cpu-saturated");
        assert_eq!(body["state"], "firing");
        assert_eq!(body["host"], "web-1");
        assert_eq!(body["value"], 93.5);
        assert_eq!(body["at"], 1_700_000_000u64, "a time is a whole number");
        assert_eq!(body.as_object().unwrap().len(), crate::alerts::HOOK_VARS.len());
    }

    #[test]
    fn chat_messages_say_the_state_and_the_host() {
        let cfg = curl_config(&target("slack"), &call(true));
        assert!(cfg.contains(r#"[FIRING] web-1: cpu-saturated 93.5% ≥ 90%"#), "{cfg}");
        let cfg = curl_config(&target("discord"), &call(false));
        assert!(cfg.contains(r#"\"content\":\"[RESOLVED] web-1:"#), "{cfg}");
        // Under a fleet the message already leads with the host.
        let mut c = call(true);
        c.message = format!("web-1: {}", c.message);
        assert!(!curl_config(&target("slack"), &c).contains("web-1: web-1"));
    }

    /// A unit alert carries the unit's last journal lines, so the page says
    /// why it failed and not only that it did.
    #[test]
    fn the_evidence_rides_along_in_every_format() {
        let mut c = call(true);
        c.detail = "api.service:\nNov 15 api[9]: bind: address in use ```".into();
        let slack: serde_json::Value =
            serde_json::from_str(&body_of(&target("slack"), &c)).unwrap();
        let text = slack["text"].as_str().unwrap();
        assert!(
            text.ends_with("```\napi.service:\nNov 15 api[9]: bind: address in use '''\n```"),
            "{text}"
        );
        assert!(body_of(&target("ntfy"), &c).ends_with("address in use ```"));
        let hook: serde_json::Value = serde_json::from_str(&webhook_body(&c)).unwrap();
        assert!(hook["detail"].as_str().unwrap().contains("address in use"));
        // Without any, the message is as it was.
        assert!(!body_of(&target("slack"), &call(true)).contains("```"));
    }

    /// The body a target is sent, read back out of its curl config.
    fn body_of(t: &NotifyTarget, c: &HookCall) -> String {
        let cfg = curl_config(t, c);
        let line = cfg.lines().find(|l| l.starts_with("data-binary = ")).unwrap();
        let quoted = &line["data-binary = ".len()..];
        serde_json::from_str::<String>(quoted).unwrap_or_else(|_| quoted.to_string())
    }

    #[test]
    fn ntfy_gets_plain_text_and_its_own_headers() {
        let cfg = curl_config(&target("ntfy"), &call(true));
        assert!(
            cfg.contains(r#"header = "Title: crabmon: cpu-saturated firing on web-1""#),
            "{cfg}"
        );
        assert!(cfg.contains(r#"header = "Priority: high""#), "{cfg}");
        assert!(cfg.contains(r#"data-binary = "cpu-saturated 93.5% ≥ 90%""#), "{cfg}");
    }

    /// Nothing in a rule name or message can break out of a quoted value or
    /// a header line, and the token goes in a header, never the URL.
    #[test]
    fn the_config_is_quoted_against_what_a_rule_can_contain() {
        let mut c = call(true);
        c.rule = "evil\"\nurl = \"https://attacker.test".into();
        c.message = "line\\one\nline two".into();
        let mut t = target("ntfy");
        t.token = "tk_secret".into();
        let cfg = curl_config(&t, &c);
        let urls: Vec<&str> = cfg.lines().filter(|l| l.starts_with("url")).collect();
        assert_eq!(urls, [r#"url = "https://example.test/hook""#], "{cfg}");
        assert!(cfg.contains(r#"data-binary = "line\\one\nline two""#), "{cfg}");
        assert!(cfg.contains(r#"header = "Authorization: Bearer tk_secret""#), "{cfg}");
        assert_eq!(cfg.lines().filter(|l| l.starts_with("header = \"Title")).count(), 1);
    }
}
