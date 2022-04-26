class ObfuscationUrlScheduler {
    // arrivalTimeSum = 0; // in seconds
    sampleRate = 0;
    estimatedRate = 0;
    periodsElapsed = 0;
    // avg = 0; // in seconds
    // preliminary_avg = 1/60; // 1/600 (10 min) by default
    
    constructor(delta, alpha, timePeriod) {
        this.delta = delta || 0;
        this.alpha = alpha || 0;
        this.timePeriod = timePeriod || 1.0; // defaults to one minute
    }

    updateArrivals() {
        this.sampleRate++;
    }

    timePeriodFunction() {
        this.periodsElapsed++;
        this.estimatedRate = ((1 - this.alpha) * this.estimatedRate) + (this.alpha * this.sampleRate);
        this.sampleRate = 0;

        console.log("Estimated rate: ", this.estimatedRate);
    }

    nextArrival() {
        var nextArrivalTime = 0;
        
        if (this.estimatedRate == 0 || this.periodsElapsed < 2) {
            return 0;
        } else {
            var actualArrivalRate = ((this.estimatedRate / this.timePeriod) < 0.1) ? 0.1 : (this.estimatedRate / this.timePeriod);

            while (nextArrivalTime < 0.5 || nextArrivalTime > 600) { // enforce an arrival time of > 30 sec, and <= 10 min
                nextArrivalTime = -Math.log(1-Math.random()) / actualArrivalRate;
            }

            // prevent too short of an arrival time

            console.log("The next arrival time (in seconds) is: " + nextArrivalTime * 60);

            return nextArrivalTime; // in minutes
        }
    }
}