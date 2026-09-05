//! Interface classification. Loopback, bridges, veth pairs and VPN tunnels are
//! marked virtual so the aggregate rate does not count the same bytes twice.

/// Prefixes treated as virtual unless the config overrides the list.
pub const DEFAULT_VIRTUAL_PREFIXES: [&str; 14] = [
    "lo", "docker", "br-", "veth", "virbr", "tun", "tap", "wg", "vmnet", "zt", "cni", "flannel",
    "kube", "utun",
];

pub fn is_virtual(name: &str, prefixes: &[String]) -> bool {
    if name == "lo" || name == "lo0" {
        return true;
    }
    prefixes.iter().any(|p| !p.is_empty() && name.starts_with(p.as_str()))
}

pub fn default_prefixes() -> Vec<String> {
    DEFAULT_VIRTUAL_PREFIXES.iter().map(|s| s.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_interfaces_on_this_host_classify_correctly() {
        let p = default_prefixes();
        for name in ["lo", "docker0", "br-ee64cceb3075", "veth3e33b66", "virbr0", "wg0"] {
            assert!(is_virtual(name, &p), "{name} should be virtual");
        }
        for name in ["wlp67s0", "eno0", "enp0s31f6", "eth0"] {
            assert!(!is_virtual(name, &p), "{name} should be physical");
        }
    }

    #[test]
    fn an_empty_prefix_list_marks_only_loopback() {
        assert!(is_virtual("lo", &[]));
        assert!(!is_virtual("docker0", &[]));
    }

    #[test]
    fn empty_prefix_strings_do_not_match_everything() {
        assert!(!is_virtual("wlp67s0", &["".to_string()]));
    }
}
