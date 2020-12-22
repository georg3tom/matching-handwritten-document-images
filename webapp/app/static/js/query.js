function showIMG(data) {
    console.log(data);
    $("#imgdiv").empty();
    var main = document.getElementById("imgdiv");


    for(var i = 0; i < data.images.length; ++i) {

        var row = document.createElement("div");
        row.className = "row";
        main.appendChild(row);
        var image = " <img src=\"/static/images/".concat(data.images[i], "\" height=\"700px\" > ");
        var col = '<div "/> ' + image + "<br> <p>" + data.images[i] + "  <br/> votes:" + data.votes[i] + "</p>" +' <div/>'
        row.innerHTML = row.innerHTML +  col;
    }
}

function submit3() {
    var query = $("#stringSearch").val();
    if(query){
        $.ajax({
            data: {
                string : query,
            },
            type : 'POST',
            url : '/searchString'
        })
            .done(function(data){
                showIMG(data);
            });
        event.preventDefault();
    }
}

function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#inpimg')
                .attr('src', e.target.result)
                .width(150)
                .height(200);
        };

        reader.readAsDataURL(input.files[0]);
    }
}

