"use strict";

var convnetjs = require("convnetjs");
var fs = require('fs');
var csv = require('csv-parser');

// error window
var Window = function(size, minsize) {
    this.v = [];
    this.size = typeof(size)==='undefined' ? 100 : size;
    this.minsize = typeof(minsize)==='undefined' ? 10 : minsize;
    this.sum = 0;
  }

Window.prototype = {
    add: function(x) {
      this.v.push(x);
      this.sum += x;
      if(this.v.length>this.size) {
        var xold = this.v.shift();
        this.sum -= xold;
      }
    },
    get_average: function() {
      if(this.v.length < this.minsize) return -1;
      else return this.sum/this.v.length;
    },
    reset: function(x) {
      this.v = [];
      this.sum = 0;
    }
}
var lossWindow = new Window(150);


// define layers
var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:13});
layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
layer_defs.push({type:'fc', num_neurons:20, activation:'tanh'});
layer_defs.push({type:'regression', num_neurons:1});
 
// create a net
var net = new convnetjs.Net();
net.makeLayers(layer_defs);
 
// train network
var trainer = new convnetjs.SGDTrainer(net, {method: 'adagrad', learning_rate: 1, l2_decay: 0.001, batch_size: 5, momentum:0.1});

var names = {"Iris-setosa" : 0,
			"Iris-versicolor": 1,
			"Iris-virginica":2}

var formatValues = function(row) {
	var x = new convnetjs.Vol([
		row["CRIM"],
		row["ZN"], 
		row["INDUS"],
		row["CHAS"], 
		row["NOX"], 
		row["RM"], 
		row["AGE"], 
		row["DIS"], 
		row["RAD"],
		row["TAX"],
		row["PTRATIO"],
		row["B"],
		row["LSTAT"]]);
	// target
	var y = row["MEDV"];
	// console.log(dict[row["class"]])
	
	return {x : x, y : y};
}

function update(){
  // forward prop the data
  
  var netx = new convnetjs.Vol(1,1,1);
  var N = df.length;
  var avloss = 0.0;

  console.log('I am in update mode');
  for(var iters=0;iters<50;iters++) {
    for(var ix=0;ix<N;ix++) {
      netx.w = df[ix];
      var stats = trainer.train(netx, label[ix]);
      avloss += stats.loss;
  		console.log(avloss)
    }
  }
  avloss /= N*iters;

}

var df = [];
var label = [];

var N_TRAIN = 400;
var k = 0;
var test_nb = 0;
var error = 0.0;
var square_error = 0.0

fs.createReadStream("data/boston.csv")
	.pipe(csv({separator: ','}))
	.on('data', function(data) {
		if (k < N_TRAIN){

			var formated = formatValues(data);

			df.push([formated.x]);
			label.push([formated.y]);

	  	var stats = trainer.train(formated.x, formated.y);
	  	lossWindow.add(stats.l2_decay_loss);
	  	console.log('loss', lossWindow.get_average())
	  	k++
	  } else {
			// Testing our model
			
			// loop our train
			//update()

			//testing
			var test = formatValues(data);
			var predicted = net.forward(test.x)
			
			
			error = predicted.w[0] - test.y;
			square_error += error * error;
			console.log('Predicted', predicted.w[0], 'Expected', test.y);
			test_nb++;
	  	}
	})
	.on("end", function(){
		console.log('Number of test', test_nb);
		console.log("MSE", square_error / test_nb);
		console.log("Finished")
	});
