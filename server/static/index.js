function clickDropDown() {
    document.getElementById("myDropdown").classList.toggle("show");
  }
  
function startLearning(){
  print('learing')
  fetch('/training',{
    method:'POST',
    mode:'no-cors',
    headers: {
      'Content-Type': 'application/json'
  },
  }).then((res)=>{
    if(res.status==200){
      console.log('Start training clients')
    }
    
  }).catch((err)=>{
    console.log(`Error ${err}`)
  })
}

function deleteUser(client_id){
  print('Delete user')
  fetch('/delete_user',{
    method:'POST',
    mode:'no-cors',
    headers: {
      'Content-Type': 'application/json'
  },
  }).then((res)=>{
    if(res.status==200){
      console.log(`Delete client with id ${client_id}`)
    }
    
  }).catch((err)=>{
    console.log(`Error ${err}`)
  })

  console.log(client_id)

  // client=document.getElementById(`${client_id}_client`)
  // client.remove()
}
  // Close the dropdown menu if the user clicks outside of it
  window.onclick = function(event) {
    if (!event.target.matches('.clients_status_training_dropbtn')) {
      var dropdowns = document.getElementsByClassName("dropdown-content");
      var i;
      for (i = 0; i < dropdowns.length; i++) {
        var openDropdown = dropdowns[i];
        if (openDropdown.classList.contains('show')) {
          openDropdown.classList.remove('show');
        }
      }
    }
  }