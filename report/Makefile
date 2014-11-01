all: main biblio tableofcontents

main: main.tex
	pdflatex main.tex

biblio: biblio.bib main.tex
	bibtex main
	pdflatex main.tex

tableofcontents: main.tex
	pdflatex main.tex

clean:
	rm *.aux *.toc *.log *.out *.bbl *.blg
