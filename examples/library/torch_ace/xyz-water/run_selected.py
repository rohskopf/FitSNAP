from subprocess import call


infiles = ['latte_cell_40.xyz', 'latte_cell_50.xyz', 'latte_cell_60.xyz']

for infile in infiles:
	call('python selected_avg.py %s' % infile,shell=True)
