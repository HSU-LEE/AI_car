# Shared setup for every .command file under Commands/.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1     # Project root, where `aicar` lives.
A=./aicar

ask() { read -r -p "$1" REPLY; echo "$REPLY"; }

finish() {
    echo
    read -r -p "Press Enter to close this window..." _
}
