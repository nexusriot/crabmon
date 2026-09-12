# fish completion for crabmon
complete -c crabmon -f

complete -c crabmon -s r -l refresh -r -d 'Refresh interval in ms (200-10000)'
complete -c crabmon -s s -l sort -r -d 'Sort column' \
    -a 'pid name cpu mem virt disk time user state threads nice'
complete -c crabmon -s a -l ascending -d 'Sort ascending'
complete -c crabmon -s d -l descending -d 'Sort descending (the default)'
complete -c crabmon -s f -l filter -r -d 'Initial process filter'
complete -c crabmon -s t -l tree -d 'Start in process-tree view'
complete -c crabmon -s l -l layout -r -d 'Initial layout' \
    -a 'dashboard processes cpu io'
complete -c crabmon -l theme -r -d 'Colour theme' \
    -a 'default mono nord solarized gruvbox'
complete -c crabmon -l no-color -d 'Disable colour'
complete -c crabmon -l no-mouse -d 'Do not capture mouse events'
complete -c crabmon -s 1 -l once -d 'Print one snapshot and exit'
complete -c crabmon -l format -r -d 'Output format for --once' -a 'json csv'
complete -c crabmon -s n -l top -r -d 'Limit --once output to N processes'
complete -c crabmon -s c -l config -r -F -d 'Alternate config file'
complete -c crabmon -s g -l group -r -d 'Group processes' \
    -a 'none service container user'
complete -c crabmon -l record -r -F -d 'Append every sample to a JSONL recording'
complete -c crabmon -l replay -r -F -d 'Replay a recording instead of sampling'
complete -c crabmon -l remote -r -d 'Monitor another host over SSH'
complete -c crabmon -l remote-command -r -d 'Command to run on the remote host'
complete -c crabmon -l serve -r -d 'Serve Prometheus metrics on ADDR'
complete -c crabmon -l stream -d 'Print one JSON snapshot per line forever'
complete -c crabmon -l diff -r -F -d 'Compare two snapshots, or one recording end to end'
complete -c crabmon -l watch -r -d 'Block until processes match QUERY, then exit 1'
complete -c crabmon -l watch-for -r -d 'Require the match to hold this many seconds'
complete -c crabmon -l watch-timeout -r -d 'Give up after this many seconds'
complete -c crabmon -s h -l help -d 'Show help'
complete -c crabmon -s V -l version -d 'Show version'
