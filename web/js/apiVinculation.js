
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

var fallecimientoFlag = 0;

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
    const final = ", por favor, indique si ¡la predicción es correcta!";
    
    fallecimientoFlag = resultado;

    document.getElementById("ResPrediccion").innerHTML = resultado? predicado+="muerto" +final : predicado+="sobrevivido"+final;
    document.getElementById("predHeader").style.display="flex";
}

function swapStuff(flagValue){ 
    
    document.getElementById("pred").style.display = flagValue ==0 ? "flex" : "none" ;
    document.getElementById("ana").style.display = flagValue ==0 ? "none"  : "flex";

    flagOut = flagValue ==0 ? "analink" : "prelink" ;
    flagTrue = flagValue==0 ? "prelink" : "analink" ;

    document.getElementById(flagTrue).classList.add("active");
    document.getElementById(flagOut).classList.remove("active");
}


function clearAll(){
    document.getElementById("edad").value="";
    document.getElementsByName("hipertensoInput")[0].checked=true;
 
    document.getElementById("eyeccion").value="";
    document.getElementById("plaquetas").value="";
    document.getElementById("creatinina").value="";
    document.getElementById("sodio").value="";

    document.getElementById("predHeader").style.display="none";
}

function savingNewData(compPred){
    var edad = document.getElementById("edad").value;
    var hip = document.getElementsByName("hipertensoInput")[0].checked;

    var hipertension = hip? 1: 0;
    var eyeccion = document.getElementById("eyeccion").value;
    var plaquetas = document.getElementById("plaquetas").value;
    var creatinina = document.getElementById("creatinina").value;
    var sodio = document.getElementById("sodio").value;

    var neoFlag=0;
    //Se murio y acertamos
    if(compPred){
        neoFlag = fallecimientoFlag? 1 : 0;
    }else{
        neoFlag = fallecimientoFlag? 0 : 1;
    }  

    console.log("Resultado = ",neoFlag);

    eel.nuevoMiembro(parseInt(edad), parseInt(hipertension),parseInt(eyeccion), parseInt(plaquetas),parseFloat(creatinina),parseInt(sodio),parseInt(compPred));

    clearAll();
}
