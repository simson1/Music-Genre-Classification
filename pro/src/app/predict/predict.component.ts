import { Component, OnInit } from '@angular/core';
import {Http} from "@angular/http";
import {
  ProbabilityPrediction,
} from "./types";
@Component({
  selector: 'predict',
  templateUrl: './predict.component.html',
  styleUrls: ['./predict.component.scss']
})
export class PredictComponent implements OnInit {
  public probabilityPredictions: ProbabilityPrediction[];
  constructor(private http: Http) { }
  
  ngOnInit() {
  }
  postMethod(files: FileList) {

    let formData = new FormData(); 
    console.log( files.item(0).name + " " + files.item(0));
    formData.append('file',files.item(0), files.item(0).name); 
    this.http.post('http://127.0.0.1:5002/api/predict', formData).subscribe((val) => {
    this.probabilityPredictions=val.json();
    });
    return false; 
    
    }

}
