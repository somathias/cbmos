# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

api:
	sphinx-apidoc -f -o source/ ../cbmos/

notebooks:
	rm -rf source/basic_example.rst source/basic_example_files/
	jupyter nbconvert --to rst ../examples/basic_example.ipynb
	mv -f ../examples/basic_example.rst ../examples/basic_example_files/ source/
	rm -rf source/convergence_example.rst source/convergence_example_files/
	jupyter nbconvert --to rst ../examples/convergence_example.ipynb
	mv -f ../examples/convergence_example.rst ../examples/convergence_example_files/ source/
	rm -rf source/generalized_events.rst source/generalized_events_files/
	jupyter nbconvert --to rst ../examples/generalized_events.ipynb
	mv -f ../examples/generalized_events.rst ../examples/generalized_events_files/ source/
