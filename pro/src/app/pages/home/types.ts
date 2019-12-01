export class ProbabilityPrediction {
    name: string;
    value: number;
}

export class SVCParameters {
    C: number = 10;
    gamma:number =0.01;
    kernel:string="rbf";
    degree:number=3;

}

export class SVCResult {
    accuracy: number;
}
