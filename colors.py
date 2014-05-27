# -*- coding: iso-8859-1 -*-

"""
Colour-blind proof distinct colours module, based on work by Paul Tol
Pieter van der Meer, 2011
SRON - Netherlands Institute for Space Research
"""
# colour palette from Paul Tol 
#  http://www.sron.nl/~pault/colourschemes.pdf
darkblue='#332288'
lightblue='#88CCEE'
teal='#44AA99'
darkgreen='#117733'
olivegreen='#999933'
khaki='#DDCC77' 
pink='#CC6677'
maroon='#882255'
magenta='#AA4499'
darkred='#661100'
slateblue='#6699CC'
darkpink='#AA4466'
darkslateblue='#4477AA'

blue=darkblue
green=darkgreen
red=darkred


colordict = {
    'darkblue':'#332288',
    'lightblue':'#88CCEE',
    'teal':'#44AA99',
    'green':'#117733',
    'olivegreen':'#999933',
    'khaki':'#DDCC77' ,
    'pink':'#CC6677',
    'maroon':'#882255',
    'magenta':'#AA4499',
    'darkred':'#661100',
    'slateblue':'#6699CC',
    'darkpink':'#AA4466',
    'darkslateblue':'#4477AA',
    }

paleblue='#809BC8'
palered='#FF6666'
paleorange='#FFCC66'
palegreen='#64C204'


# colour table in HTML hex format
hexcols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', 
           '#CC6677', '#882255', '#AA4499', '#661100', '#6699CC', '#AA4466',
           '#4477AA']

greysafecols = ['#809BC8', '#FF6666', '#FFCC66', '#64C204']

xarr = [[12], 
        [12, 6], 
        [12, 6, 5], 
        [12, 6, 5, 3], 
        [0, 1, 3, 5, 6], 
        [0, 1, 3, 5, 6, 8], 
        [0, 1, 2, 3, 5, 6, 8], 
        [0, 1, 2, 3, 4, 5, 6, 8], 
        [0, 1, 2, 3, 4, 5, 6, 7, 8], 
        [0, 1, 2, 3, 4, 5, 9, 6, 7, 8], 
        [0, 10, 1, 2, 3, 4, 5, 9, 6, 7, 8], 
        [0, 10, 1, 2, 3, 4, 5, 9, 6, 11, 7, 8]]


def getColorByName( colorname ) : 
    """ get the hex code for one of the colors in the palette
    by providing the name """
    if colorname not in colordict : 
        print("You must provide one of the 13 color names")
        return(None) 
    return( colordict[colorname] )

def getColorByHex( hexcode ) : 
    """ get the name for one of the colors in the palette
    by providing the hex code """
    if hexcode not in colordict.values() : 
        print("You must provide one of the 13 color hex codes")
        return(None) 
    for key in colordict : 
        if colordict[key] == hexcode : return( key )


# get specified nr of distinct colours in HTML hex format.
# in: nr - number of colours [1..12]
# returns: list of distinct colours in HTML hex
def get_distinct(nr, returnNames=False):

    #
    # check if nr is in correct range
    #
    
    if nr < 1 or nr > 12:
        print "wrong nr of distinct colours!"
        return

    #
    # get list of indices
    #
    
    lst = xarr[nr-1]
    
    #
    # generate colour list by stepping through indices and looking them up
    # in the colour table
    #

    i_col = 0
    col = [0] * nr
    for idx in lst:
        col[i_col] = hexcols[idx]
        i_col+=1
    if returnNames : 
        colnames = [ getColorByHex( c ) for c in col ]
        return( colnames )
    return col

# gets 4 colours, which also look distinct in black&white
# returns: list of 4 colours in 
#def get_distinct_grey():
    
# displays usage information and produces example plot.
if __name__ == '__main__':
    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    print __doc__
    print "usage examples: "
    print "print distinct_colours.get_distinct(2)"
    print get_distinct(2)
    print "print distinct_colours.greysafecols"
    print greysafecols

    print "\ngenerating example plot: distinct_colours_example.png"
    plt.close()
    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2*np.pi*t)
    c = np.cos(2*np.pi*t)
    cols = get_distinct(2)
    plt.plot(t, s, linewidth=1.0, c=cols[0])
    plt.plot(t, c, linewidth=1.0, c=cols[1])

    plt.xlabel('time (s)')
    plt.ylabel('voltage (mV)')
    plt.title('Distinct colours example')
    plt.grid(True)
    plt.show()
    plt.savefig("distinct_colours_example.png")
