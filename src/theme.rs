//! Colour themes. Every colour the UI draws comes from here, so `--no-color`
//! and user overrides are a single substitution rather than a grep-and-replace.

use std::collections::BTreeMap;

use ratatui::style::{Color, Modifier, Style};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Theme {
    pub name: String,
    pub ok: Color,
    pub warn: Color,
    pub crit: Color,
    pub accent: Color,
    pub border: Color,
    pub title: Color,
    pub text: Color,
    pub dim: Color,
    pub sel_fg: Color,
    pub sel_bg: Color,
    /// Cycled per CPU core / per network interface series.
    pub series: Vec<Color>,
}

impl Default for Theme {
    fn default() -> Self {
        Theme::preset("default").expect("the default preset always exists")
    }
}

pub const PRESETS: [&str; 5] = ["default", "mono", "nord", "solarized", "gruvbox"];

impl Theme {
    pub fn preset(name: &str) -> Option<Theme> {
        let t = match name {
            "default" => Theme {
                name: "default".into(),
                ok: Color::Green,
                warn: Color::Yellow,
                crit: Color::Red,
                accent: Color::Cyan,
                border: Color::DarkGray,
                title: Color::White,
                text: Color::Reset,
                dim: Color::DarkGray,
                sel_fg: Color::Black,
                sel_bg: Color::Cyan,
                series: vec![
                    Color::Cyan,
                    Color::Green,
                    Color::Yellow,
                    Color::Magenta,
                    Color::Blue,
                    Color::LightRed,
                ],
            },
            "mono" => Theme {
                name: "mono".into(),
                ok: Color::Reset,
                warn: Color::Reset,
                crit: Color::Reset,
                accent: Color::Reset,
                border: Color::Reset,
                title: Color::Reset,
                text: Color::Reset,
                dim: Color::Reset,
                sel_fg: Color::Reset,
                sel_bg: Color::Reset,
                series: vec![Color::Reset],
            },
            "nord" => Theme {
                name: "nord".into(),
                ok: Color::Rgb(163, 190, 140),
                warn: Color::Rgb(235, 203, 139),
                crit: Color::Rgb(191, 97, 106),
                accent: Color::Rgb(136, 192, 208),
                border: Color::Rgb(76, 86, 106),
                title: Color::Rgb(236, 239, 244),
                text: Color::Rgb(216, 222, 233),
                dim: Color::Rgb(76, 86, 106),
                sel_fg: Color::Rgb(46, 52, 64),
                sel_bg: Color::Rgb(136, 192, 208),
                series: vec![
                    Color::Rgb(136, 192, 208),
                    Color::Rgb(163, 190, 140),
                    Color::Rgb(235, 203, 139),
                    Color::Rgb(180, 142, 173),
                    Color::Rgb(129, 161, 193),
                ],
            },
            "solarized" => Theme {
                name: "solarized".into(),
                ok: Color::Rgb(133, 153, 0),
                warn: Color::Rgb(181, 137, 0),
                crit: Color::Rgb(220, 50, 47),
                accent: Color::Rgb(42, 161, 152),
                border: Color::Rgb(88, 110, 117),
                title: Color::Rgb(238, 232, 213),
                text: Color::Rgb(147, 161, 161),
                dim: Color::Rgb(88, 110, 117),
                sel_fg: Color::Rgb(0, 43, 54),
                sel_bg: Color::Rgb(42, 161, 152),
                series: vec![
                    Color::Rgb(42, 161, 152),
                    Color::Rgb(133, 153, 0),
                    Color::Rgb(181, 137, 0),
                    Color::Rgb(211, 54, 130),
                    Color::Rgb(38, 139, 210),
                ],
            },
            "gruvbox" => Theme {
                name: "gruvbox".into(),
                ok: Color::Rgb(184, 187, 38),
                warn: Color::Rgb(250, 189, 47),
                crit: Color::Rgb(251, 73, 52),
                accent: Color::Rgb(131, 165, 152),
                border: Color::Rgb(80, 73, 69),
                title: Color::Rgb(235, 219, 178),
                text: Color::Rgb(213, 196, 161),
                dim: Color::Rgb(124, 111, 100),
                sel_fg: Color::Rgb(40, 40, 40),
                sel_bg: Color::Rgb(131, 165, 152),
                series: vec![
                    Color::Rgb(131, 165, 152),
                    Color::Rgb(184, 187, 38),
                    Color::Rgb(250, 189, 47),
                    Color::Rgb(211, 134, 155),
                    Color::Rgb(142, 192, 124),
                ],
            },
            _ => return None,
        };
        Some(t)
    }

    /// Apply `[colors]` overrides from the config file. Unknown keys and
    /// unparseable values are ignored so a stale config never breaks startup.
    pub fn with_overrides(mut self, overrides: &BTreeMap<String, String>) -> Self {
        for (key, value) in overrides {
            let Some(color) = parse_color(value) else {
                continue;
            };
            match key.as_str() {
                "ok" => self.ok = color,
                "warn" => self.warn = color,
                "crit" => self.crit = color,
                "accent" => self.accent = color,
                "border" => self.border = color,
                "title" => self.title = color,
                "text" => self.text = color,
                "dim" => self.dim = color,
                "sel_fg" => self.sel_fg = color,
                "sel_bg" => self.sel_bg = color,
                _ => {}
            }
        }
        self
    }

    /// How a selected row is painted.
    ///
    /// `mono` (and `--no-color`) leave both selection colours at `Reset`, which
    /// used to make the highlight bold-only — invisible on any terminal that
    /// renders bold as plain. Reversing the cell is the one way to mark a row
    /// without using a colour.
    pub fn selection(&self) -> Style {
        if self.sel_fg == Color::Reset && self.sel_bg == Color::Reset {
            return Style::default().add_modifier(Modifier::REVERSED | Modifier::BOLD);
        }
        Style::default().fg(self.sel_fg).bg(self.sel_bg).add_modifier(Modifier::BOLD)
    }

    pub fn series_at(&self, i: usize) -> Color {
        if self.series.is_empty() {
            Color::Reset
        } else {
            self.series[i % self.series.len()]
        }
    }

    /// Green → yellow → red as a 0..1 ratio climbs past `warn`/`crit`.
    pub fn usage(&self, ratio: f64, warn: f64, crit: f64) -> Color {
        if ratio >= crit {
            self.crit
        } else if ratio >= warn {
            self.warn
        } else {
            self.ok
        }
    }

    /// Colour a temperature relative to its critical threshold, with absolute
    /// fallbacks when the sensor reports no critical point.
    pub fn temp(&self, temp: f32, critical: Option<f32>, warn: f64, crit: f64) -> Color {
        match critical {
            Some(c) if c > 0.0 => self.usage((temp / c) as f64, warn, crit),
            _ => {
                if temp >= 85.0 {
                    self.crit
                } else if temp >= 70.0 {
                    self.warn
                } else {
                    self.ok
                }
            }
        }
    }
}

/// Accepts `red`, `lightblue`, `#ff8800`, `12` (256-colour index) and `reset`.
pub fn parse_color(s: &str) -> Option<Color> {
    let s = s.trim().to_ascii_lowercase();
    if let Some(hex) = s.strip_prefix('#') {
        // `len()` is bytes but the slices below are byte ranges, so a 6-byte
        // value whose first char is multi-byte ("#日本") passed the guard and
        // then panicked on a char boundary — taking the whole session down from
        // a typo in `[colors]`.
        if hex.len() != 6 || !hex.is_ascii() {
            return None;
        }
        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
        return Some(Color::Rgb(r, g, b));
    }
    if let Ok(idx) = s.parse::<u8>() {
        return Some(Color::Indexed(idx));
    }
    let c = match s.as_str() {
        "reset" | "default" => Color::Reset,
        "black" => Color::Black,
        "red" => Color::Red,
        "green" => Color::Green,
        "yellow" => Color::Yellow,
        "blue" => Color::Blue,
        "magenta" => Color::Magenta,
        "cyan" => Color::Cyan,
        "gray" | "grey" => Color::Gray,
        "darkgray" | "darkgrey" => Color::DarkGray,
        "lightred" => Color::LightRed,
        "lightgreen" => Color::LightGreen,
        "lightyellow" => Color::LightYellow,
        "lightblue" => Color::LightBlue,
        "lightmagenta" => Color::LightMagenta,
        "lightcyan" => Color::LightCyan,
        "white" => Color::White,
        _ => return None,
    };
    Some(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_advertised_preset_resolves() {
        for name in PRESETS {
            assert!(Theme::preset(name).is_some(), "{name} missing");
        }
        assert!(Theme::preset("nope").is_none());
    }

    #[test]
    fn mono_preset_paints_nothing() {
        let t = Theme::preset("mono").unwrap();
        assert_eq!(t.crit, Color::Reset);
        assert_eq!(t.usage(1.0, 0.7, 0.9), Color::Reset);
        assert_eq!(t.temp(200.0, Some(100.0), 0.75, 0.9), Color::Reset);
    }

    #[test]
    fn a_colourless_theme_still_marks_the_selected_row() {
        // Bold alone is not a highlight: plenty of terminals render it as plain.
        let mono = Theme::preset("mono").unwrap().selection();
        assert!(mono.add_modifier.contains(Modifier::REVERSED), "{mono:?}");

        let default = Theme::default().selection();
        assert!(!default.add_modifier.contains(Modifier::REVERSED));
        assert_eq!(default.bg, Some(Theme::default().sel_bg));
    }

    #[test]
    fn colors_parse_from_names_hex_and_indexes() {
        assert_eq!(parse_color("Red"), Some(Color::Red));
        assert_eq!(parse_color("#ff8800"), Some(Color::Rgb(255, 136, 0)));
        assert_eq!(parse_color("33"), Some(Color::Indexed(33)));
        assert_eq!(parse_color("#fff"), None);
        assert_eq!(parse_color("chartreuse"), None);
    }

    #[test]
    fn overrides_apply_and_bad_values_are_ignored() {
        let mut o = BTreeMap::new();
        o.insert("crit".to_string(), "#ff0000".to_string());
        o.insert("warn".to_string(), "not-a-color".to_string());
        o.insert("bogus_key".to_string(), "red".to_string());
        let base = Theme::default();
        let t = base.clone().with_overrides(&o);
        assert_eq!(t.crit, Color::Rgb(255, 0, 0));
        assert_eq!(t.warn, base.warn);
    }

    #[test]
    fn usage_thresholds_step_at_the_right_ratios() {
        let t = Theme::default();
        assert_eq!(t.usage(0.69, 0.7, 0.9), t.ok);
        assert_eq!(t.usage(0.70, 0.7, 0.9), t.warn);
        assert_eq!(t.usage(0.95, 0.7, 0.9), t.crit);
    }

    #[test]
    fn temperature_falls_back_to_absolutes_without_a_critical_point() {
        let t = Theme::default();
        assert_eq!(t.temp(60.0, None, 0.75, 0.9), t.ok);
        assert_eq!(t.temp(75.0, None, 0.75, 0.9), t.warn);
        assert_eq!(t.temp(90.0, None, 0.75, 0.9), t.crit);
        assert_eq!(t.temp(105.0, Some(110.0), 0.75, 0.9), t.crit);
    }

    #[test]
    fn series_wraps_and_survives_an_empty_palette() {
        let t = Theme::default();
        assert_eq!(t.series_at(0), t.series_at(t.series.len()));
        let empty = Theme { series: vec![], ..Theme::default() };
        assert_eq!(empty.series_at(3), Color::Reset);
    }

    /// The `#rrggbb` guard counted bytes and the parse sliced bytes, so a
    /// 6-byte value whose first char is multi-byte got through and panicked on
    /// a char boundary — killing the session over a typo in `[colors]`.
    #[test]
    fn a_non_ascii_colour_value_is_rejected_rather_than_panicking() {
        for bad in ["#日本", "#😀ab", "#ＡＢＣ", "#ff88ää"] {
            assert_eq!(parse_color(bad), None, "{bad} was accepted");
        }
        assert_eq!(parse_color("#ff8800"), Some(Color::Rgb(255, 136, 0)));
    }
}
