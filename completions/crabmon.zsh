#compdef crabmon

_crabmon() {
    _arguments -s \
        '(-r --refresh)'{-r,--refresh}'[refresh interval in ms]:milliseconds:' \
        '(-s --sort)'{-s,--sort}'[sort column]:key:(pid name cpu mem virt disk time user state threads nice)' \
        '(-g --group)'{-g,--group}'[group processes]:by:(none service container user)' \
        '--record[append every sample to a JSONL recording]:file:_files' \
        '--replay[replay a recording instead of sampling]:file:_files' \
        '--remote[monitor another host over SSH]:target:_hosts' \
        '--remote-command[command to run on the remote host]:command:' \
        '--serve[serve Prometheus metrics on ADDR]:address:' \
        '(-a --ascending)'{-a,--ascending}'[sort ascending]' \
        '(-f --filter)'{-f,--filter}'[initial process filter]:query:' \
        '(-t --tree)'{-t,--tree}'[start in tree view]' \
        '(-l --layout)'{-l,--layout}'[initial layout]:layout:(dashboard processes cpu io)' \
        '--theme[colour theme]:theme:(default mono nord solarized gruvbox)' \
        '--no-color[disable colour]' \
        '--no-mouse[do not capture mouse events]' \
        '(-1 --once)'{-1,--once}'[print one snapshot and exit]' \
        '--format[output format for --once]:format:(json csv)' \
        '(-n --top)'{-n,--top}'[limit --once output to N processes]:count:' \
        '(-c --config)'{-c,--config}'[alternate config file]:file:_files' \
        '(-h --help)'{-h,--help}'[show help]' \
        '(-V --version)'{-V,--version}'[show version]'
}

_crabmon "$@"
