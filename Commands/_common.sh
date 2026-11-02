# Commands/ 안의 모든 .command 파일이 불러오는 공통 부분
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1     # 프로젝트 루트 (aicar가 있는 곳)
A=./aicar

ask() { read -r -p "$1" REPLY; echo "$REPLY"; }

finish() {
    echo
    read -r -p "엔터를 누르면 창을 닫습니다..." _
}
