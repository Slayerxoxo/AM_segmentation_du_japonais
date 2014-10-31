all: main tableofcontents

main: main.tex
	pdflatex main.tex

tableofcontents: main.tex
	pdflatex main.tex

clean:
	rm *.aux *.toc *.log *.out
