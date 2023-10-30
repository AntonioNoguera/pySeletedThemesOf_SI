
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





function swapStuff(flagValue){ 
    
    document.getElementById("pred").style.display = flagValue ==0 ? "flex" : "none" ;
    document.getElementById("ana").style.display = flagValue ==0 ? "none"  : "flex";

    flagOut = flagValue ==0 ? "analink" : "prelink" ;
    flagTrue = flagValue==0 ? "prelink" : "analink" ;

    document.getElementById(flagTrue).classList.add("active");
    document.getElementById(flagOut).classList.remove("active");
}