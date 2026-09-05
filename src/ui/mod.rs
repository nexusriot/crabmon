//! Rendering. Panels draw from `App` and nothing else, so the whole dashboard
//! can be rendered onto a `TestBackend` and asserted on. The process table is
//! the one panel that takes `&mut App`: it records the geometry it drew so a
//! later mouse click can be mapped back to a row or a column header.

pub mod cpu;
pub mod disks;
pub mod gpu;
pub mod header;
pub mod mem;
pub mod net;
pub mod popups;
pub mod power;
pub mod procs;
pub mod psi;
pub mod sensors;

use ratatui::layout::{Constraint, Direction, Layout as RLayout, Rect};
use ratatui::style::Style;
use ratatui::widgets::{Block, Borders};
use ratatui::Frame;
use serde::{Deserialize, Serialize};

use crate::app::{App, Mode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Layout {
    /// Everything at once: CPU, memory, processes, and the right-hand column.
    Dashboard,
    /// Process table using the full width.
    Processes,
    /// Per-core detail.
    Cpu,
    /// Network, disks and GPU.
    Io,
}

impl Layout {
    pub fn name(self) -> &'static str {
        match self {
            Layout::Dashboard => "dashboard",
            Layout::Processes => "processes",
            Layout::Cpu => "cpu",
            Layout::Io => "io",
        }
    }

    pub fn parse(s: &str) -> Option<Layout> {
        match s.trim().to_ascii_lowercase().as_str() {
            "dashboard" | "dash" => Some(Layout::Dashboard),
            "processes" | "procs" => Some(Layout::Processes),
            "cpu" => Some(Layout::Cpu),
            "io" => Some(Layout::Io),
            _ => None,
        }
    }
}

pub fn block<'a>(app: &App, title: impl Into<String>) -> Block<'a> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(ratatui::text::Span::styled(title.into(), Style::default().fg(app.theme.title)))
}

/// Split a rect into vertical slices, dropping any that would be zero-height.
fn rows(area: Rect, constraints: Vec<Constraint>) -> Vec<Rect> {
    RLayout::default().direction(Direction::Vertical).constraints(constraints).split(area).to_vec()
}

pub fn draw(f: &mut Frame<'_>, app: &mut App) {
    let size = f.size();
    if size.width < 20 || size.height < 6 {
        // Too small for any panel; say so rather than panicking on a layout.
        f.render_widget(ratatui::widgets::Paragraph::new("terminal too small"), size);
        return;
    }

    let header_h = if app.cfg.panels.header { 3 } else { 0 };
    let status_h = 1;
    let chunks = rows(
        size,
        vec![Constraint::Length(header_h), Constraint::Min(3), Constraint::Length(status_h)],
    );
    if app.cfg.panels.header {
        header::draw(f, chunks[0], app);
    }
    match app.cfg.layout {
        Layout::Dashboard => draw_dashboard(f, chunks[1], app),
        Layout::Processes => procs::draw(f, chunks[1], app),
        Layout::Cpu => cpu::draw_detail(f, chunks[1], app),
        Layout::Io => draw_io(f, chunks[1], app),
    }
    header::draw_status(f, chunks[2], app);

    match app.mode {
        Mode::SignalMenu => popups::signal_menu(f, size, app),
        Mode::ConfirmKill => popups::confirm_kill(f, size, app),
        Mode::Renice => popups::renice(f, size, app),
        Mode::Affinity => popups::affinity(f, size, app),
        Mode::Detail => popups::detail(f, size, app),
        Mode::Help => popups::help(f, size, app),
        Mode::Alerts => popups::alerts(f, size, app),
        Mode::AuditLog => popups::audit_log(f, size, app),
        Mode::Normal | Mode::Filter => {}
    }
}

fn draw_dashboard(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let cols = RLayout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    let mut left = Vec::new();
    if app.cfg.panels.cpu {
        left.push(Constraint::Length(8));
    }
    if app.cfg.panels.mem {
        left.push(Constraint::Length(3));
        left.push(Constraint::Length(3));
    }
    left.push(Constraint::Min(5));
    let slots = rows(cols[0], left);

    let mut i = 0;
    if app.cfg.panels.cpu {
        cpu::draw(f, slots[i], app);
        i += 1;
    }
    if app.cfg.panels.mem {
        mem::draw_memory(f, slots[i], app);
        mem::draw_swap(f, slots[i + 1], app);
        i += 2;
    }
    if app.cfg.panels.procs {
        procs::draw(f, slots[i], app);
    }

    draw_side_column(f, cols[1], app);
}

/// The right-hand column, ordered by `panel_order` in the config.
fn draw_side_column(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let mut panels: Vec<(&str, Constraint)> = Vec::new();
    for name in app.cfg.panel_order.clone() {
        let entry = match name.as_str() {
            "net" if app.cfg.panels.net => ("net", Constraint::Length(7)),
            "sensors" if app.cfg.panels.sensors => ("sensors", Constraint::Length(7)),
            "disks" if app.cfg.panels.disks => ("disks", Constraint::Min(4)),
            "gpu" if app.cfg.panels.gpu && !app.snap.gpus.is_empty() => {
                ("gpu", Constraint::Length(app.snap.gpus.len() as u16 + 3))
            }
            "cgroup" if app.cfg.panels.cgroup && app.snap.cgroup.is_some() => {
                ("cgroup", Constraint::Length(4))
            }
            "psi" if app.cfg.panels.psi && app.snap.psi.is_available() => {
                ("psi", Constraint::Length(6))
            }
            "power" if app.cfg.panels.power && app.snap.power.is_available() => {
                ("power", Constraint::Length(app.snap.power.batteries.len().max(1) as u16 + 3))
            }
            _ => continue,
        };
        panels.push(entry);
    }
    if panels.is_empty() {
        return;
    }
    let slots = rows(area, panels.iter().map(|(_, c)| *c).collect());
    for ((name, _), slot) in panels.iter().zip(slots) {
        match *name {
            "net" => net::draw(f, slot, app),
            "sensors" => sensors::draw(f, slot, app),
            "disks" => disks::draw(f, slot, app),
            "gpu" => gpu::draw(f, slot, app),
            "cgroup" => mem::draw_cgroup(f, slot, app),
            "psi" => psi::draw(f, slot, app),
            "power" => power::draw(f, slot, app),
            _ => {}
        }
    }
}

fn draw_io(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let gpu_h = if app.cfg.panels.gpu && !app.snap.gpus.is_empty() {
        app.snap.gpus.len() as u16 + 3
    } else {
        0
    };
    let slots =
        rows(area, vec![Constraint::Percentage(50), Constraint::Min(4), Constraint::Length(gpu_h)]);
    net::draw_detail(f, slots[0], app);
    disks::draw(f, slots[1], app);
    if gpu_h > 0 {
        gpu::draw(f, slots[2], app);
    }
}

/// Centred popup rect, clamped so it always fits on screen.
pub fn centered_rect(pct_x: u16, pct_y: u16, area: Rect) -> Rect {
    let w = (area.width * pct_x / 100).clamp(10, area.width);
    let h = (area.height * pct_y / 100).clamp(3, area.height);
    Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_names_round_trip() {
        for l in [Layout::Dashboard, Layout::Processes, Layout::Cpu, Layout::Io] {
            assert_eq!(Layout::parse(l.name()), Some(l));
        }
        assert_eq!(Layout::parse("nope"), None);
    }

    #[test]
    fn cycling_layouts_forward_then_back_returns_to_the_start() {
        for l in [Layout::Dashboard, Layout::Processes, Layout::Cpu, Layout::Io] {
            assert_eq!(l.next().prev(), l);
        }
    }

    #[test]
    fn popups_stay_inside_tiny_terminals() {
        let area = Rect { x: 0, y: 0, width: 12, height: 5 };
        let p = centered_rect(80, 60, area);
        assert!(p.x + p.width <= area.width);
        assert!(p.y + p.height <= area.height);
        assert!(p.width >= 10 && p.height >= 3);
    }
}
