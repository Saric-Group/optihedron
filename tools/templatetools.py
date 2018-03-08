import os

def fillTemplate(template, filledTemplate, placeHolder, filledPlaceHolder):
	with open(template) as templateFile:
		with open(filledTemplate, 'w') as filledTemplateFile:
			for line in templateFile:
				filledTemplateFile.write(line.replace(placeHolder, filledPlaceHolder))
	os.remove(template)
	os.rename(filledTemplate, template)

def fill(content,placeHolder,filledPlaceHolder):
	content.replace(placeHolder, filledPlaceHolder)
	return content