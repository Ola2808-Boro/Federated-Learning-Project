
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

function deleteUser($event){
  rows_arr=[]
  console.log('Delete User');
  const re = /\d/;
  table=document.getElementById('clients')
  rows=table.getElementsByTagName('tr')
  for(i=0;i<rows.length;i++){
    if (rows[i].getElementsByTagName('td').length!=0){
      console.log(rows[i],rows[i].getElementsByTagName('td'))
      id=rows[i].getElementsByTagName('td')[0].innerHTML
      rows_arr.push({
        'index':i,
        'id':id
      })
    }
  }

  row_delete=re.exec($event.innerHTML)[0]
  fetch(`/delete_user/${row_delete}`,{
    method:'POST',
    mode:'no-cors',
    headers: {
      'Content-Type': 'application/json'
  },
  }).then((res)=>{
    if(res.status==200){
      rows_arr.forEach(element => {
        console.log('id',element.id,'row',row_delete)
        if(element.id==row_delete){
          table.deleteRow(element.index)
          console.log(`Delete client with id ${re.exec($event.innerHTML)}`)
        }
      });
    }  
  }).catch((err)=>{
    console.log(`Error ${err}`)
  })
}


function startServer(event){

  event.preventDefault();
  const lr=document.getElementById('lr_server').value;
  const epochs=document.getElementById('epochs_server').value;
  const batch_size=document.getElementById('batch_size_server').value;
  const optim=document.getElementById('optim_server').value;

  fetch(`/server/${lr}/${epochs}/${batch_size}/${optim}`,{
    method:'POST',
    mode:'no-cors',
    headers: {
      'Content-Type': 'application/json'
  },
  }).then((res)=>{
    if(res.status==200){
      console.log('Amazing')
    }  
  }).catch((err)=>{
    console.log(`Error ${err}`)
  })
}

function addClient(event){

    event.preventDefault();
    const client_name=document.getElementById('name_client').value;
    const lr=document.getElementById('lr_client').value;
    const epochs=document.getElementById('epochs_client').value;
    const batch_size=document.getElementById('batch_size_client').value;
    const optim=document.getElementById('optim_client').value;
    console.log(`Parameters ${client_name}, ${lr}, ${epochs},${batch_size}, ${optim}`);
    fetch(`/client/${client_name}/${lr}/${epochs}/${batch_size}/${optim}`,{
      method:'POST',
      mode:'no-cors',
      headers: {
        'Content-Type': 'application/json'
    },
    }).then((res)=>{
      if(res.status==200){
        console.log('Amazing')
      }  
    }).catch((err)=>{
      console.log(`Error ${err}`)
    })
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

