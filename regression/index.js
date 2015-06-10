"use strict";

var convnetjs = require("convnetjs");
var fs = require('fs');
var csv = require('csv-parser');


// define layers
var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:12});
// layer_defs.push({type:'regression', num_neurons: 5});
layer_defs.push({type:'regression', num_neurons: 2});
 
// create a net
var net = new convnetjs.Net();
net.makeLayers(layer_defs);
 
// train network
var trainer = new convnetjs.Trainer(net, {method: 'adadelta', l2_decay: 0.001, batch_size: 10});

var formatValues = function(row) {
	var x = new convnetjs.Vol(1, 1 ,12 , 0.0); // a 1x1x2 volume initialized to 0's.
	x.w[0] = parseFloat(row["CRIM"]);
	x.w[1] = parseFloat(row["ZN"]);
	x.w[2] = parseFloat(row["CHAS"]);
	x.w[3] = parseFloat(row["NOX"]);
	x.w[4] = parseFloat(row["RM"]);
	x.w[5] = parseFloat(row["AGE"]);
	x.w[6] = parseFloat(row["DIS"]);
	x.w[7] = parseFloat(row["RAD"]);
	x.w[8] = parseFloat(row["PTRATIO"]);
	x.w[9] = parseFloat(row["B"]);
	x.w[10] = parseFloat(row["LSTAT"]);
	x.w[11] = parseFloat(row["INDUS"]);

	var y = [parseFloat(row["TAX"]), parseFloat(row["MEDV"])];
	return {x : x, y : y};
}

fs.createReadStream("data/boston.csv")
	.pipe(csv({separator: ','}))
	.on('data', function(data) {
		var formated = formatValues(data);
  		trainer.train(formated.x, formated.y);
	})
	.on("end", function(){
		var row = { "CRIM": 0.10959,
		  "ZN": 0,
		  "INDUS": 11.93,
		  "CHAS": 0,
		  "NOX": 0.573,
		  "RM": 6.794,
		  "AGE": 89.3,
		  "DIS": 2.3889,
		  "RAD": 1,
		  "TAX": 273,
		  "PTRATIO": 21,
		  "B": 393.45,
		  "LSTAT": 6.48,
		  "MEDV": 22 };

		var test = formatValues(row);
		var predicted = net.forward(test.x)
		console.log(predicted, test.y)

	})



