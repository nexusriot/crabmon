# bash completion for crabmon
_crabmon() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="-r --refresh -s --sort -a --ascending -d --descending -f --filter
          -t --tree -l --layout --theme --no-color --no-mouse -1 --once --format
          -n --top -c --config -g --group --record --replay --remote
          --remote-command --serve --stream --diff --watch --watch-for
          --watch-timeout -h --help -V --version"

    case "$prev" in
        -s|--sort)
            COMPREPLY=($(compgen -W "pid name cpu mem virt disk time user state threads nice" -- "$cur"))
            return 0 ;;
        -g|--group)
            COMPREPLY=($(compgen -W "none service container user" -- "$cur"))
            return 0 ;;
        --record|--replay|--diff)
            COMPREPLY=($(compgen -f -- "$cur"))
            return 0 ;;
        --watch|--watch-for|--watch-timeout)
            return 0 ;;
        --remote|--remote-command|--serve)
            return 0 ;;
        -l|--layout)
            COMPREPLY=($(compgen -W "dashboard processes cpu io" -- "$cur"))
            return 0 ;;
        --theme)
            COMPREPLY=($(compgen -W "default mono nord solarized gruvbox" -- "$cur"))
            return 0 ;;
        --format)
            COMPREPLY=($(compgen -W "json csv" -- "$cur"))
            return 0 ;;
        -c|--config)
            COMPREPLY=($(compgen -f -- "$cur"))
            return 0 ;;
        -r|--refresh|-n|--top|-f|--filter)
            return 0 ;;
    esac

    COMPREPLY=($(compgen -W "$opts" -- "$cur"))
}
complete -F _crabmon crabmon
