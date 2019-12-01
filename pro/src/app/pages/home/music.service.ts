import {Injectable} from '@angular/core';
import {Http} from "@angular/http";
import {Observable} from "rxjs/Observable";
import 'rxjs/add/operator/map';
import {
    SVCParameters,
    SVCResult
} from "./types";
@Injectable()
export class MusicService {

    constructor(private http: Http) {
    }

    public trainModel(svcParameters: SVCParameters): Observable<SVCResult> {
        console.log(svcParameters);
        return this.http.post('http://127.0.0.1:5002/api/train', svcParameters).map((res) => res.json());
    }
}
