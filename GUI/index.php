<!DOCTYPE html>
<html>

    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">

    <!-- jQuery library -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
    
    <!-- Latest compiled JavaScript -->
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
    
    <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Raleway">
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
        <link href="https://fonts.googleapis.com/css?family=Lato" rel="stylesheet">

        
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css">
<!-- Bootstrap core CSS -->
<link href="css/bootstrap.min.css" rel="stylesheet">
<!-- Material Design Bootstrap -->
<link href="css/mdb.min.css" rel="stylesheet">
<!-- Your custom styles (optional) -->
<link href="css/style.css" rel="stylesheet">
<!--Import Google Icon Font-->
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
  <!--Import materialize.css-->
  <link type="text/css" rel="stylesheet" href="css/materialize.min.css"  media="screen,projection"/>

  <!--Let browser know website is optimized for mobile-->
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>

      
<link href="https://fonts.googleapis.com/css?family=Rokkitt" rel="stylesheet">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">

<!-- jQuery library -->
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>

    <!-- Compiled and minified CSS -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js"></script>


<head>
    
    
    
    
    
            
    
  <title>Content Based Image Retreival</title>
  <style type="text/css">
     
  body {
      text-align: center;
      margin-left:auto;
      margin-right:auto;
      font-family: Lato; 
      color: black;
  }
  
  .header {
    color: black;
      text-align: center;
    font-size: 40px;
  }

    form {
      text-align: center;
    }
  </style>
</head>



<body>
     <h1 class="header">CONTENT BASED IMAGE RETREIVAL</h1>
     

     <button type="button" class="btn btn-primary" data-toggle="modal" data-target="#imagemodal" >
        Image Search
      </button>      
  
      <button type="button" class="btn btn-primary" data-toggle="modal" data-target="#textmodal"  >
          Text Search
        </button>

     <div class="modal fade" id="textmodal"   >
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title">TEXT SEARCH</h5>
            <button type="button" class="close" data-dismiss="modal" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
          </div>
          <div class="modal-body">
                  <div class="row">
                  <form class="col s12" action="search.php" method="post" enctype="multipart/form-data">
                    <div class="row">
                      <div class="input-field col s6">
                        <i class="material-icons prefix">search</i>
                        <textarea id="icon_prefix2" type="text" name="searchterm" class="materialize-textarea"></textarea>
                    <label for="icon_prefix2">Search Query</label>
                  
                  
              <button class="btn btn-primary" id="submit2" type="submit" name="submit2">Search
                  <i class="material-icons right">send</i>
              </button>
                  </form>
                  </div>
                  </div>
                  </div>
                  
          </div>
        </div>
      </div>
    </div>








    <div class="modal fade" id="imagemodal" tabindex="-1" role="dialog" >
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title">IMAGE SEARCH</h5>
            <button type="button" class="close" data-dismiss="modal" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
          </div>
          <div class="modal-body">


              <form action="upload.php" method="post" enctype="multipart/form-data">
                <input type='file' id="fileToUpload" name="fileToUpload" accept=".jpg, .png, .jpeg"; />        
                <button class="btn btn-primary" id="submit" type="submit" style="text-align: center" name="submit" action="submit">Upload & Search
                    <i class="material-icons right">send</i>
                </button>
              </form>

            


          </div>
          </div>
      </div>
    </div>

   
  <script type="text/javascript" src="js/jquery-3.3.1.min.js"></script>
    <!-- Bootstrap tooltips -->
    <script type="text/javascript" src="js/popper.min.js"></script>
    <!-- Bootstrap core JavaScript -->
    <script type="text/javascript" src="js/bootstrap.min.js"></script>
    <!-- MDB core JavaScript -->
    <script type="text/javascript" src="js/mdb.min.js"></script>
 <script type="text/javascript" src="js/materialize.min.js"></script>


</body>
</html>