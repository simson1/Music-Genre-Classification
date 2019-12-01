// external modules
import {BrowserModule} from '@angular/platform-browser';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import {NgModule} from '@angular/core';
import {PreloadAllModules, RouterModule} from "@angular/router";
import 'hammerjs';

// own modules and components
import {AppComponent} from './app.component';
import {SharedModule} from "./shared/shared.module";
import {ROUTES} from "./app.routes";
import {HomeComponent} from "./pages/home/home.component";
import {MusicService} from "./pages/home/music.service";
import { PredictComponent } from './predict/predict.component';

@NgModule({
    declarations: [
        AppComponent,
        HomeComponent,
        PredictComponent,
    ],
    imports: [
        BrowserModule,
        BrowserAnimationsModule,
        RouterModule.forRoot(ROUTES, {useHash: false, preloadingStrategy: PreloadAllModules}),
        SharedModule
    ],
    providers: [MusicService],
    bootstrap: [AppComponent]
})
export class AppModule {
}
