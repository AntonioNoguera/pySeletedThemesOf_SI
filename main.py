import eel

eel.init('web',allowed_extensions=['.js','.html','.css'])

@eel.expose
def saludar():
    print("tilin arrived") 
    
eel.start('index.html', size=(100, 500))