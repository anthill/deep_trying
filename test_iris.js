"use strict";

var convnetjs = require("convnetjs");
var fs = require('fs');
var csv = require('csv-parser');


// testing list of target
var unique_y = {};
var distinct_Label = [];
var df = [];
var labels = [];
var avloss = 0.0;

var net = new convnetjs.Net();

// define layers
var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:4});
layer_defs.push({type:'fc', num_neurons:5, activation:'relu'});
layer_defs.push({type:'softmax', num_classes:3});
 
// create a net
net.makeLayers(layer_defs);
 
// train network
var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.1, l2_decay:0.001});

var formatValues = function(row) {
	var x = new convnetjs.Vol(1, 1 , 4);

	// input
	x.w[0] = row["sepal_length"];
	x.w[1] = row["sepal_width"];
	x.w[2] = row["petal_length"];
	x.w[3] = row["petal_width"];

	// target
	var y = row["class"];
		
	if (unique_y[y] !== 0) {
		distinct_Label.push(y);
		unique_y[y] = 0;
	}
	return {x : x, y : y};
}

fs.createReadStream("data/iris.csv")
	.pipe(csv({separator: ','}))
	.on('data', function(data) {
		var formated = formatValues(data);
  		var stats = trainer.train(formated.x, formated.y);
  		//console.log('formated', formated.x)
  		avloss += stats.loss;
  		//console.log('predicted value: ' + net.forward(formated.x).w[0]);
  		//console.log('avloss', avloss)
  		df.push(formated.x);
  		labels.push(formated.y);
	})
	.on("end", function(){
		// Testing our model
		var row = { "sepal_length": 5.9,
					"sepal_width": 3.0,
					"petal_length": 5.1,
					"petal_width": 1.8};

		var test = formatValues(row);
		var predicted = net.forward(test.x)

		//console.log("prediction : ", predicted, test.y)
		
		
		// find the max label proba
		var max = -Infinity
		for (var i = 0; i < predicted.w.length; i++) {
			if (max < predicted.w[i]) {
				max = predicted.w[i];
				var max_id = i;
			}
		}

		for (var i = 0; i < distinct_Label.length; i++) {
			if (distinct_Label[i] === 'Iris-virginica') {
				var result_list = i;
			} 
		}
		//console.log('df', df.keys())
		console.log("final result for our taget (Iris-virginica) is ", predicted.w[result_list])
		console.log("Best proba is", distinct_Label[max_id], "with", max);

		// console.log('Df', df.length)

		var netx = new convnetjs.Vol(1,1,4);
		
		var trainer2 = new convnetjs.SGDTrainer(net, {learning_rate:0.01, l2_decay:0.001});

		var N = df.length;

		for(var iters=0;iters<50;iters++) {
		    for(var ix=0;ix<N;ix++) {
		    	//console.log("df[ix]", df[ix])
		    	netx.w = df[ix];

		    	var stats = trainer2.train(netx, labels[ix]);
				avloss += stats.loss;
		    }
		}
		//console.log('netx', netx)

		avloss /= N*iters;
		//console.log('avloss', avloss)

	})
