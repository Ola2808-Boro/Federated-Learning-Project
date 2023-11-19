// inputs in forms
// const validate_input=(id)=>{
//   console.log('Validate')
//   input=document.getElementById(id);
//   console.log(input)
//   input.addEventListener("input", (event) => {
//     if (input.validity.typeMismatch) {
//       input.setCustomValidity("I am expecting",input.getAttribute('type'));
//     }   
//   });
//   console.log('Type',input.getAttribute('type'))
// }
// id=['lr_server','lr_client','epochs_server','epochs_client','batch_size_server','batch_size_client','optim_server','optim_client','name_client']
// for(i=0;i<id.length;i++){

//   validate_input(id[i])
// }

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

function enableSubmit_client(){
  
  const client_name=document.getElementById('name_client').value;
  const lr=document.getElementById('lr_client').value;
  const epochs=document.getElementById('epochs_client').value;
  const batch_size=document.getElementById('batch_size_client').value;
  const optim=document.getElementById('optim_client').value;

  if(client_name.trim!=="" && lr.trim()!=="" && epochs.trim()!=="" && batch_size.trim()!=="" && optim.trim()!==""){
    console.log('Alert',optim.trim().toLowerCase())
    if(optim.trim().toLowerCase()!=="sgd" && optim.trim().toLowerCase()!=="adam"){
      alert('Change optim')
    }
    else{
      btn=document.getElementById('client_start')
      btn.disabled=false;
    }
  }
}

function enableSubmit_server(){

  const lr=document.getElementById('lr_server').value;
  const epochs=document.getElementById('epochs_server').value;
  const batch_size=document.getElementById('batch_size_server').value;
  const optim=document.getElementById('optim_server').value;
  console.log(`Lr ${lr.trim()} epochs ${epochs.trim()} batch-size ${batch_size.trim()} optim ${optim.trim()}`)
  if(lr.trim()!=="" && epochs.trim()!=="" && batch_size.trim()!=="" && optim.trim()!==""){
    if(optim.trim().toLowerCase()!=="sgd" && optim.trim().toLowerCase()!=="adam"){
      alert('Change optim')
      console.log('Alert')
    }
    else{
      btn=document.getElementById('server_start')
      btn.disabled=false;
    }
  }
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

