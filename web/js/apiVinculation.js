
console.log("he llegao")

//Vinculation API

//Graficas de Tiempo
function displayStuffTest(){
    eel.graficas();
}

//Gráficas de Distribución
function displayDistribution(){
    eel.distribucion();
}

//Correlation
function correlation(){
    eel.corr();
}

function displayAccuracy(){
    eel.verificarRed(false);
}

function prediction(){
    var edad = document.getElementById("edad").value;
    var hip = document.getElementsByName("hipertensoInput")[0].checked;

    var hipertension = hip? 1: 0;
    var eyeccion = document.getElementById("eyeccion").value;
    var plaquetas = document.getElementById("plaquetas").value;
    var creatinina = document.getElementById("creatinina").value;
    var sodio = document.getElementById("sodio").value;
    
    console.log(edad+" - "+hipertension+" - "+ eyeccion+" - "+plaquetas+" - "+creatinina+" - "+sodio);
    
    eel.prediccion(parseInt(edad), parseInt(hipertension),parseInt(eyeccion), parseInt(plaquetas),parseFloat(creatinina),parseInt(sodio))(callback);
    
}

function callback(resultado){
    console.log("Callback llamado");
    console.log(resultado);
    var predicado= "La predicción indica que el paciente a ";

    document.getElementById("ResPrediccion").innerHTML = resultado? predicado+="muerto." : predicado+="sobrevivido.";
}




function swapStuff(flagValue){ 
    
    document.getElementById("pred").style.display = flagValue ==0 ? "flex" : "none" ;
    document.getElementById("ana").style.display = flagValue ==0 ? "none"  : "flex";

    flagOut = flagValue ==0 ? "analink" : "prelink" ;
    flagTrue = flagValue==0 ? "prelink" : "analink" ;

    document.getElementById(flagTrue).classList.add("active");
    document.getElementById(flagOut).classList.remove("active");
}