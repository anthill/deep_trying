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
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:4});
layer_defs.push({type: 'softmax', num_classes: 3});
 
// create a net
var net = new convnetjs.Net();
net.makeLayers(layer_defs);
 
// train network
var trainer = new convnetjs.Trainer(net, {momentum: 0.1, l2_decay: 0.001});

var names = {"Iris-setosa" : 0,
			"Iris-versicolor": 1,
			"Iris-virginica":2}

var formatValues = function(row) {
	var x = new convnetjs.Vol([
		row["sepal_length"], 
		row["sepal_width"], 
		row["petal_length"], 
		row["petal_width"]]);
	
	// target
	var y = names[row["class"]];
	// console.log(dict[row["class"]])
	
	return {x : x, y : y};
}

var N_TRAIN = 120;
var k = 0;

fs.createReadStream("data/iris.csv")
	.pipe(csv({separator: ','}))
	.on('data', function(data) {
		if (k < N_TRAIN){

			var formated = formatValues(data);

	  		var stats = trainer.train(formated.x, formated.y);
	  		lossWindow.add(stats.l2_decay_loss);
	  		console.log('loss', lossWindow.get_average())
	  		k++
	  	} else {
			// Testing our model

			var test = formatValues(data);
			var predicted = net.forward(test.x)
			
			
			// find the max label proba
			var max = -Infinity
			for (var i = 0; i < predicted.w.length; i++) {
				if (max < predicted.w[i]) {
					max = predicted.w[i];
					var max_id = i;
				}
			};
	
			console.log("Predicted", max_id, "expected", test.y);

	  	}
	})
	.on("end", function(){
		console.log("Finished")
	});
