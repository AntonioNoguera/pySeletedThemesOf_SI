
console.log("he llegao")
function displayStuffTest(){
    eel.graficas();
}

function swapStuff(flagValue){ 
    
    document.getElementById("pred").style.display = flagValue ==0 ? "flex" : "none" ;
    document.getElementById("ana").style.display = flagValue ==0 ? "none"  : "flex";

    flagOut = flagValue ==0 ? "analink" : "prelink" ;
    flagTrue = flagValue==0 ? "prelink" : "analink" ;

    document.getElementById(flagTrue).classList.add("active");
    document.getElementById(flagOut).classList.remove("active");
}