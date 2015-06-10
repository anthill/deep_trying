"use strict";

var convnetjs = require("convnetjs");
var fs = require('fs');
var csv = require('csv-parser');


//// PARAMETERS

var N_TRAIN = 400;
var ITER = 500;

// define layers
var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:12});
layer_defs.push({type:'fc', num_neurons:20, activation:'tanh'});
layer_defs.push({type:'fc', num_neurons:40, activation:'relu'});
layer_defs.push({type:'fc', num_neurons:20, activation:'sigmoid'});
layer_defs.push({type:'regression', num_neurons:2});


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

// create a net
var net = new convnetjs.Net();
net.makeLayers(layer_defs);
 
// train network
// var trainer = new convnetjs.SGDTrainer(net, {method: 'adagrad', learning_rate: 0.001, l2_decay: 0.001, batch_size: 3, momentum:0.0});
var trainer = new convnetjs.Trainer(net, {method: 'adagrad', l2_decay: 0.001,
                                    			batch_size: 1, learning_rate: 0.01,
                                    			momentum:0.1});

var formatValues = function(row) {
	var x = new convnetjs.Vol([
		parseFloat(row["ZN"]),
		parseFloat(row["RM"]), 
		parseFloat(row["DIS"]),
		parseFloat(row["CRIM"]),
		parseFloat(row["NOX"]), 
		parseFloat(row["PTRATIO"]),
		parseFloat(row["TAX"]),
		parseFloat(row["AGE"]), 
		parseFloat(row["B"]),
		parseFloat(row["INDUS"]),
		parseFloat(row["CHAS"]), 
		parseFloat(row["RAD"])
		]);
	// target
	var y = [parseFloat(row["MEDV"]),
			parseFloat(row["LSTAT"]) 
			]; // this must be a list

	return {x : x, y : y};
}


var train = [];
var test = [];

var lossWindow = new Window(N_TRAIN);
var square_error = 0.0

fs.createReadStream("data/boston.csv")
	.pipe(csv({separator: ','}))
	.on('data', function(data) {
	// Train only in steaming
		var formated = formatValues(data);
		if ( Math.random() < 0.80) {
			train.push(formated);
			var stats = trainer.train(formated.x, formated.y);
			lossWindow.add(stats.loss);
	  	} else {
	  		test.push(formated);
	  	}

	  	
	})
	.on("end", function(){

		//testing

		test.forEach(function(v){
			var predicted = net.forward(v.x);
			var error = (predicted.w[0] - v.y[0]) + (predicted.w[1] - v.y[1]) ;
			square_error += error * error;
			console.log('Predicted', predicted.w[0], 'et', predicted.w[1], 'Expected', v.y[0], 'et', v.y[1]);
		})
		console.log("MSE", square_error / test.length);
		console.log("Finished")
		console.log("Taille du train :", train.length);
		console.log("Taille du test :", test.length)
	});
