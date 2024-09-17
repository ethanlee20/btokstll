
# Generate a standard image name.

# Note: delta C9 integer values will be converted
#       to equivalent decimal values 
#       (i.e. 2 -> 2.0)


shopt -s extglob


int_to_float() {
    local int=$1
    float="${int}.0"
    echo "$float"
}


dc9_real=$1
trial=$2

case ${dc9_real#-} in
    *([0-9])  ) dc9_real=$(int_to_float $dc9_real) ;;
    *         ) ;;
esac

echo "dc9_${dc9_real}_${trial}"