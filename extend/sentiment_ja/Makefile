
all:
	make prepare
	make install
	make example

prepare:
	pip3 install virtualenv
	virtualenv venv

install:
	( . venv/bin/activate; venv/bin/python setup.py install)

example:
	( . venv/bin/activate; venv/bin/python test/1.py )
