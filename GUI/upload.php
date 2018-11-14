<?php

header("Cache-Control: no-store, no-cache, must-revalidate, max-age=0");
header("Cache-Control: post-check=0, pre-check=0", false);
header("Pragma: no-cache");

if(isset($_POST['submit'])) {
	$file = $_FILES['fileToUpload'];
	$fileName = $_FILES['fileToUpload']['name'];
	$fileTmpName = $_FILES['fileToUpload']['tmp_name'];
	$fileSize = $_FILES['fileToUpload']['size'];
	$fileError = $_FILES['fileToUpload']['error'];
	$fileType = $_FILES['fileToUpload']['type'];

	$fileExt = explode('.', $fileName);
	$fileActualExt = strtolower(end($fileExt));

	$allowed = array('jpg','jpeg','png');
	if (in_array($fileActualExt, $allowed)) {
		if($fileError === 0) {
			$fileNameNew = "file.jpg";
			$fileDestination = 'uploads/'.$fileNameNew;
			move_uploaded_file($fileTmpName, $fileDestination);
			header("Location: result.php");
		}else {
			echo "Error!";
		}
	}else {
		echo "Invalid image!";
	}


	
	//$result = exec("C:\\Users\\John\\Anaconda3\\python.exe C:\\Users\\John\\Desktop\\PredScript.py");
	//$resultArray = json_decode($result);
	//foreach ($resultArray as $row) {
	//	echo $row . "<BR>";
	//}
}
?>