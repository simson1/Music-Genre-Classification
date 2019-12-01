import {Routes} from "@angular/router";
import {HomeComponent} from "./pages/home/home.component";
import { PredictComponent } from "./predict/predict.component";

export const ROUTES: Routes = [
    // routes from pages
    {path: 'home', component: HomeComponent, data: {title: 'Home'}},
    {path: 'predict', component: PredictComponent, data: {title: 'predict'}},

    // default redirect
    {path: '**', redirectTo: '/home'}

];
