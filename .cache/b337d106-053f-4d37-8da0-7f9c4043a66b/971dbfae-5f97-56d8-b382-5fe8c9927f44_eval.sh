FIND_FILE_1="/etc/vim/vimrc"
FIND_FILE_2="$HOME/.vimrc"
FIND_STR="set number"

if grep -q "$FIND_STR" "$FIND_FILE_1"; then
    echo "The File Has Set Number!"
    exit 0
fi

if grep -q "$FIND_STR" "$FIND_FILE_2"; then
    echo "The File Has Set Number!"
    exit 0
else
    echo "The File Has Not Set Number!"
    exit 1
fi