//! Copying to the clipboard over OSC 52.
//!
//! The terminal itself does the copying, so this works through SSH and tmux
//! with no X11, no Wayland and no helper binary — which is the whole point when
//! the process you want to paste about is on a server.

use std::io::Write;

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Base64, hand-rolled to keep the dependency list short.
pub fn base64(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Terminals cap OSC 52 payloads; anything longer is silently dropped by many
/// of them, so crabmon truncates rather than appearing to do nothing.
pub const MAX_PAYLOAD: usize = 8192;

/// The escape sequence that puts `text` on the system clipboard.
pub fn osc52(text: &str) -> String {
    let mut bytes = text.as_bytes();
    if bytes.len() > MAX_PAYLOAD {
        bytes = &bytes[..MAX_PAYLOAD];
    }
    format!("\x1b]52;c;{}\x07", base64(bytes))
}

/// Write the sequence straight to the terminal, bypassing the frame buffer.
pub fn copy(text: &str) -> std::io::Result<()> {
    let mut out = std::io::stdout();
    out.write_all(osc52(text).as_bytes())?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_the_rfc_test_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_handles_bytes_above_ascii() {
        assert_eq!(base64("é".as_bytes()), "w6k=");
        assert_eq!(base64(&[0xff, 0xff, 0xff]), "////");
        assert_eq!(base64(&[0, 0, 0]), "AAAA");
    }

    #[test]
    fn the_sequence_is_wrapped_for_the_system_clipboard() {
        let seq = osc52("hi");
        assert!(seq.starts_with("\x1b]52;c;"), "{seq:?}");
        assert!(seq.ends_with('\x07'));
        assert!(seq.contains("aGk="));
    }

    #[test]
    fn oversized_payloads_are_truncated_rather_than_dropped_by_the_terminal() {
        let huge = "x".repeat(MAX_PAYLOAD * 2);
        let seq = osc52(&huge);
        // 8192 bytes encode to 10924 base64 characters plus the wrapper.
        assert!(seq.len() < MAX_PAYLOAD * 2, "payload was not truncated");
        assert!(seq.starts_with("\x1b]52;c;"));
    }

    #[test]
    fn every_byte_length_encodes_to_a_multiple_of_four() {
        for n in 0..64 {
            let data = vec![b'a'; n];
            assert_eq!(base64(&data).len() % 4, 0, "length {n}");
        }
    }
}
