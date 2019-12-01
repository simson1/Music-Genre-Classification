import {Component, OnInit} from '@angular/core';
import { MusicService } from "./music.service";
import {
    ProbabilityPrediction,
    SVCParameters,
    SVCResult
} from "./types";

@Component({
    selector: 'home',
    templateUrl: './home.component.html',
    styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

    public svcParameters: SVCParameters = new SVCParameters();
    public svcResult: SVCResult;
    public probabilityPredictions: ProbabilityPrediction[];
    constructor(private MusicService: MusicService) {
    }

    ngOnInit() {
    }

    public trainModel() {
        this.MusicService.trainModel(this.svcParameters).subscribe((svcResult) => {
            console.log(svcResult);
            this.svcResult = svcResult;
        });
    }
}
