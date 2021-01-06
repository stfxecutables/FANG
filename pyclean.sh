PROJECT="/home/derek/Desktop/genetic/code"
PYTHON="$HOME/.pyenv/versions/3.8.5/bin/python3.8" # change this to the correct python for you
SRC="/home/derek/Desktop/genetic/code/src"
TST="/home/derek/Desktop/genetic/code/test"

eval cd $PROJECT
echo 'Fixing `src` with autoflake...' && $PYTHON -m autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r $SRC
echo 'Fixing `test` with autoflake...' && $PYTHON -m autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r $TST
echo 'Formatting imports with isort...' && $PYTHON -m isort .
echo 'Formatting code with black...' && $PYTHON -m black .
echo 'Linting code with flake8...' && $PYTHON -m flake8 .
echo 'Type checking with mypy...' && find . -iname "*.py" -exec $PYTHON -m mypy {} +
