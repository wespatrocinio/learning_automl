FILE=bossa_nova.csv
if [ -f "$FILE" ]; then
    echo "$FILE already exists"
else
    echo "dowloading $FILE"
    curl -O -L https://raw.githubusercontent.com/wespatrocinio/music_genre_classification/master/data/lyrics/bossa_nova.csv
fi

FILE=funk.csv
if [ -f "$FILE" ]; then
    echo "$FILE already exists"
else
    echo "dowloading $FILE"
    curl -O -L https://raw.githubusercontent.com/wespatrocinio/music_genre_classification/master/data/lyrics/funk.csv
fi

FILE=gospel.csv
if [ -f "$FILE" ]; then
    echo "$FILE already exists"
else
    echo "dowloading $FILE"
    curl -O -L https://raw.githubusercontent.com/wespatrocinio/music_genre_classification/master/data/lyrics/gospel.csv
fi


FILE=sertanejo.csv
if [ -f "$FILE" ]; then
    echo "$FILE already exists"
else
    echo "dowloading $FILE"
    curl -O -L https://raw.githubusercontent.com/wespatrocinio/music_genre_classification/master/data/lyrics/sertanejo.csv
fi