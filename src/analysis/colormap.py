stepsize_colors = {
    'GTD2adagrad': 'blue',
    'GTD2amsgrad': 'green',
    'GTD2schedule': 'black',
    'GTD2': 'orange',

    'TDCadagrad': 'blue',
    'TDCamsgrad': 'green',
    'TDCschedule': 'black',
    'TDC': 'orange',

    'HTDadagrad': 'blue',
    'HTDamsgrad': 'green',
    'HTDschedule': 'black',
    'HTD': 'orange',

    'TDadagrad': 'blue',
    'TDamsgrad': 'green',
    'TDschedule': 'black',
    'TD': 'orange',

    'LSTD': 'red',
}

colors = {
    'HTD': '#984ea3',
    'GTD2': 'grey',
    'TDC': '#4daf4a',
    'TD': '#377eb8',
    'ReghTDC': '#ff7f00',
    'REGH_TDC': '#ff7f00',
    'ReghGTD2': '#f781bf',
    'VTrace': '#e41a1c',
    'VTRACE': '#e41a1c',
    'VTraceGTD2': '#a65628',
    'TDRCC': 'red',

    'LSTD': 'black',
}

# handle duplicates due to stepsize algorithms
stepsizes = ['adagrad', 'amsgrad', 'schedule']
# double iterate keys since the dict dynamically grows during outer iteration
for key in [k for k in colors]:
    color = colors[key]
    for ss in stepsizes:
        colors[key + ss] = color
