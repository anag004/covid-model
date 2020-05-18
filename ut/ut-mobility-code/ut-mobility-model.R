library(rstan)
library(dplyr)
library(ggplot2)
library(lubridate)
library(lme4)
library(rstanarm)
library(directlabels)
library(cowplot)
library(stringr)

## For running parallel chains in Stan
## options(mc.cores = min(6, parallel::detectCores()))

options(mc.cores = parallel::detectCores())

theme_set(theme_half_open())

cumulativeDeaths2 <- function(ypost, df) {

  ## if (!all(df == arrange(df, state, date))) {
  ##   stop("Dataframe must be arranged by state and date")
  ## }

  ## Posterior draws for deaths on days in the observed dataset become
  ## fixed to the observed value

  before_after <- (df$date <= today() - 1)


  for (j in 1:ncol(ypost)) {
    if (before_after[j] == TRUE) ypost[, j] <- df$deaths[j]
  }

  ## Row indices of states
  state_rows <- df %>% 
    group_by(state) %>%
    group_rows()

  ypostcm <- matrix(nrow = nrow(ypost), ncol = ncol(ypost)) 

  ## Cumulative sum for each state, for each posterior draw
  for (i in 1:nrow(ypost)) {
    for (k in 1:length(state_rows)) {
      ypostcm[i, state_rows[[k]]] <- cumsum(ypost[i, state_rows[[k]]])
    }
  }

  list(ypost = ypost, ypostcm = ypostcm)
}

## Cleans up Safegraph social distancing data
source("sdmetrics-explore-covariates-sunmon.R")

sessionInfo()

###############################################################################
                                        #              Data prep              #
###############################################################################

source("nyt-jhu-state.R")

us_states <- jhu

###############################################################################
                                        #     Social distancing predictors    #
###############################################################################

## Only 50 states + DC
us_states <- us_states %>%
  filter(state %in% state_abbr$Name) %>%
  mutate(date = ymd(date))

glimpse(us_states)
glimpse(state_both)

## Compare end dates
max(us_states$date)
max(state_both$date)

## Create a new df with covariates
us_states_covariates <- us_states %>%
  left_join(state_both %>% dplyr::select(-state_short)) %>%
  arrange(state, date) 

glimpse(us_states_covariates)

## Any missing data?
any(is.na(us_states_covariates))




###############################################################################
                                        #               Run Stan              #
###############################################################################



nburn <- 2000
niter <- 2000


glm3stan <- stan_glmer.nb(
  deaths ~
    poly(days_since_thresh, 2) * (PC1_lag_pc + PC2_lag_pc + PC3_lag_pc +
                                  rural_urban + weekend) +
    (poly(days_since_thresh, 2) | state),
  offset = log(population), data=us_states_covariates,
  chains = 3, iter = nburn + niter, warmup = nburn)


cat("Posterior predictive sampling...\n")

ypost3 <- posterior_predict(glm3stan, newdata = us_states_covariates)

save(list=ls(), file = sprintf("model_fits/stan-intermediate-%s.Rdata", now()))


###############################################################################
                                        #   Posterior predictive for future   #
###############################################################################

## Vector of states
all_states <- unique(us_states$state)

## No. days into the future
moreDays <- 7 * 3 + 1

us_futureList <- vector("list", moreDays)

## Create dataframe of future dates for all state
for (i in 1:length(all_states)) {

  stateLast <- us_states %>%
    arrange(state, date) %>% 
    filter(state == all_states[i]) %>%
    tail(1)

  futureI <- NULL

  for (j in 1:moreDays) {
    futureI <- rbind(futureI,
                     stateLast %>%
                     mutate(date = date + j,
                            days_since_thresh = days_since_thresh + j)) 
      
  }

  us_futureList[[i]] <- futureI

}

us_future <- plyr::rbind.fill(us_futureList) %>%
  arrange(state, date)

## Add covariates to these dates
us_future_covariates <- us_future %>%
  left_join(state_both %>% dplyr::select(-state_short)) %>%
  arrange(state, date)

us_future_covariates %>% glimpse()

us_future_covariates_w0 <- us_future_covariates %>%
  mutate(weekend = 0)

us_future_covariates_w1 <- us_future_covariates %>%
  mutate(weekend = 1)

## If any covariates in future are missing, extrapolate from past
if (any(is.na(us_future_covariates$lag_dist_traveled))) {
  for (i in 1:nrow(us_future_covariates)) {
    if (is.na(us_future_covariates$lag_dist_traveled[i])) {
      us_future_covariates[i, names(us_future_covariates)[-(1:8)]] <-
        us_future_covariates[i - 1, names(us_future_covariates)[-(1:8)]]
    }
  }
}

us_future_covariates$date %>% min()

## Add these future values to existing data
us_states_oose <- rbind(us_states_covariates, us_future_covariates) %>%
  arrange(state, date)

us_states_oosew0 <- rbind(us_states_covariates, us_future_covariates_w0) %>%
  arrange(state, date)

us_states_oosew1 <- rbind(us_states_covariates, us_future_covariates_w1) %>%
  arrange(state, date) 


cat("Posterior predictive sampling for forecast...\n")


ypost_oose3w0 <- posterior_predict(glm3stan, newdata = us_states_oosew0)
ypost_oose3w1 <- posterior_predict(glm3stan, newdata = us_states_oosew1)

ypost_oose3 <- (5/7) * ypost_oose3w0 + (2/7) * ypost_oose3w1

## yhat_oose3 <- colMeans(ypost_oose3)
yhat_oose3 <- apply(ypost_oose3, 2, quantile, probs = 0.5)
ylo_oose3 <- apply(ypost_oose3, 2, quantile, probs = 0.05)
yhi_oose3 <- apply(ypost_oose3, 2, quantile, probs = 0.95)

cat("\nDone with posterior predictive sampling....\n")

## Cumulative deaths
## ypostcm_oose1 <- cumulativeDeaths(ypost_oose1, us_states_oose)

ypostcm_oose3 <- cumulativeDeaths2(ypost_oose3, us_states_oose)$ypostcm

## yhatcm_oose3 <- apply(ypostcm_oose3, 3, mean)
yhatcm_oose3 <- apply(ypostcm_oose3, 2, quantile, 0.5)
ylocm_oose3 <- apply(ypostcm_oose3, 2, quantile, 0.05)
yhicm_oose3 <- apply(ypostcm_oose3, 2, quantile, 0.95)

###############################################################################
                                        #      Data frames of predictions     #
###############################################################################

us_states_ooseNA <- rbind(us_states_covariates,
                          us_future_covariates %>%
                          mutate(deaths = NA,
                               cumulative_deaths = NA)) %>% 
  arrange(state, date) 



us_fit_oose3 = mutate(us_states_ooseNA,
                      ## deaths_hat = ifelse(date <= today() - 1, NA, yhat_oose3),
                      deaths_hat = yhat_oose3,
                      deaths_lo = ifelse(date <= today() - 1, NA, ylo_oose3),
                      deaths_hi = ifelse(date <= today() - 1, NA, yhi_oose3),
                      deathscm_hat = yhatcm_oose3,
                      deathscm_lo = ylocm_oose3,
                      deathscm_hi = yhicm_oose3
                      ) 

us_fit_oose3_NoNA = mutate(us_states_ooseNA,
                      ## deaths_hat = ifelse(date <= today() - 1, NA, yhat_oose3),
                           ## deaths_hat = ifelse(date <= today() - 1, deaths, yhat_oose3),
                           deaths_hat = yhat_oose3,
                      deaths_lo = ifelse(date <= today() - 1, deaths, ylo_oose3),
                      deaths_hi = ifelse(date <= today() - 1, deaths, yhi_oose3),
                      deathscm_hat = yhatcm_oose3,
                      deathscm_lo = ylocm_oose3,
                      deathscm_hi = yhicm_oose3
                      ) 


###############################################################################
                                        #      Estimate peaks for states      #
###############################################################################

cat("Sampling the linear predictor for the peaks...\n")

ylin_forecast3w0 <- posterior_linpred(glm3stan, newdata = us_states_oosew0)
ylin_forecast3w1 <- posterior_linpred(glm3stan, newdata = us_states_oosew1)

ylin_forecast3 <- (5/7) * ylin_forecast3w0 + (2/7) * ylin_forecast3w1

stateRows <- us_states_oose %>%
  group_by(state) %>%
  group_rows()

statePeakDf <- data.frame(state = rep(NA, length(stateRows)),
                          prob_peak_already = NA,
                          prob_peak_07days = NA,
                          prob_peak_10days = NA,
                          prob_peak_14days = NA)

stateDfList <- vector("list", length(stateRows))
stateLinList <- vector("list", length(stateRows))

for (k in 1:length(stateRows)) {

  ## Subset of rows (columns of posterior matrix) for this state
  stateRowsK <- stateRows[[k]]

  ## Subset of posterior for linear predictor for this state

  stateSamplesK <- ylin_forecast3[, stateRowsK]

  ## Dates
  datesK <- us_states_oose[stateRowsK, ]$date

  ## Vector of peak dates
  peakDateVec <- datesK[apply(stateSamplesK, 1, which.max)]



  ## Calculate Monte Carlo probabilities
  statePeakDf[k, ] <- c(us_states_oose$state[stateRowsK[[1]]],
                        min(mean(peakDateVec <= today()) %>% round(3), 0.999),
                        min(mean(peakDateVec <= today() + 7) %>% round(3), 0.999),
                        min(mean(peakDateVec <= today() + 10) %>% round(3), 0.999),
                        min(mean(peakDateVec <= today() + 14) %>% round(3), 0.999))
  
  cat(sprintf("State %i out of %i...\n", k, length(stateRows)))
}

statePeakDf %>% glimpse()

###############################################################################
                                        #        Estimate peak for USA        #
###############################################################################

Dates <- us_states_oose %>%
  distinct(date) %>%
  pull(date)

dateRows <- vector("list", length(Dates))

for (i in 1:length(Dates)) {
  dateRows[[i]] <- which(us_states_oose$date == Dates[i])
}

USAlin <- matrix(nrow = nrow(ylin_forecast3), ncol = length(Dates))

for (k in 1:length(Dates)) {
  if (length(dateRows[[k]]) == 1) {
    USAlin[, k] <- exp(ylin_forecast3[, dateRows[[k]]])
  } else {
    USAlin[, k] <- apply(ylin_forecast3[, dateRows[[k]]],
                         1, function(x) sum(exp(x)))
  }
}

USAlin <- log(USAlin)

USApeakDates <- Dates[apply(USAlin, 1, which.max)]

USApeakDf <- data.frame(
  date = Dates,
  prob_peak_already = mean(USApeakDates <= today()) %>% round(3),
  prob_peak_07days = mean(USApeakDates <= today() + 7) %>% round(3),
  prob_peak_10days = mean(USApeakDates <= today() + 10) %>% round(3),
  prob_peak_14days = mean(USApeakDates <= today() + 14) %>% round(3)
  ) %>%
  arrange(date)

###############################################################################
                                        #           State spaghetti           #
###############################################################################

ypost_oose_fix3 <- cumulativeDeaths2(ypost_oose3, us_states_oose)$ypost

stateRows <- us_states_oose %>%
  group_by(state) %>%
  group_rows()

stateDfList <- vector("list", length(stateRows))

for (k in 1:length(stateRows)) {

    ## Subset of rows (columns of posterior matrix) for this state
  stateRowsK <- stateRows[[k]]

  StateK <- us_states_oose$state[stateRowsK[1]]

  ## Subset of posterior for linear predictor for this state

  stateSamplesK <- ypost_oose_fix3[, stateRowsK] 
  stateSamplesCmK <- ypostcm_oose3[, stateRowsK] 

  sampleQuantiles <- stateSamplesCmK[, length(stateRowsK)] %>%
    quantile(probs = seq(0.1, 0.9, length.out = 10))

  ## Rows <- sample(1:nrow(stateSamplesCmK), 9)

  Rows <- rep(NA, length(sampleQuantiles))


  DatesK <- us_states_oose$date[stateRowsK]

  stateDfListK <- vector("list", length(Rows))

  for (j in 1:length(Rows)) {
    Rows[j] <- which.min(abs(stateSamplesCmK[, length(stateRowsK)] -
                             sampleQuantiles[j]))

    stateDfListK[[j]] <- data.frame(
      state = StateK,
      date = DatesK,
      daily_deaths = stateSamplesK[Rows[j], ],
      cumulative_deaths = stateSamplesCmK[Rows[j], ],
      chain = j
    ) %>%
      mutate(date = ymd(date))
    
  }

  stateDfList[[k]] <- plyr::rbind.fill(stateDfListK)

  ## Dates
  datesK <- us_states_oose[stateRowsK, ]$date

  ## Vector of peak dates
  peakDateVec <- datesK[apply(stateSamplesK, 1, which.max)]

  ## Calculate Monte Carlo probabilities
  statePeakDf[k, ] <- c(us_states_oose$state[stateRowsK[[1]]],
                        mean(peakDateVec <= today()) %>% round(4),
                        mean(peakDateVec <= today() + 7) %>% round(4),
                        mean(peakDateVec <= today() + 10) %>% round(4),
                        mean(peakDateVec <= today() + 14) %>% round(4))
  
  cat(sprintf("State %i out of %i...\n", k, length(stateRows)))
}

stateSpagDf <- stateDfList %>% plyr::rbind.fill() 

stateSpagDfWide <- stateSpagDf %>%
  pivot_wider(names_from = chain,
              values_from = c(daily_deaths, cumulative_deaths),
              names_prefix = c("chain")) 

cat("Writing spaghetti file....\n")

readr::write_csv(stateSpagDfWide,
                 sprintf("../forecasts/archive/UT-COVID19-statespaghetti-%s.csv", today()))

readr::write_csv(stateSpagDfWide,
                 sprintf("../forecasts/UT-COVID19-statespaghetti-latest.csv"))

###############################################################################
                                        #             USA forecast            #
###############################################################################

all_states <- us_states %>% pull(state) %>% unique()
states_yesterday <- us_states %>% filter(date == today() - 1) %>% pull(state)

US_deathsactual <- us_states_ooseNA %>%
  group_by(date) %>%
  summarize(US_deaths = sum(deaths, na.rm = TRUE)) 

US_deathsactual <- US_deathsactual %>%
  mutate(US_deaths = ifelse(date >= "2020-04-01" & as.integer(US_deaths) == 0,
                            NA,
                            US_deaths))


dateRows <- us_states_oose %>%
  group_by(date) %>%
  group_rows()

US_deaths_date <- rep(NA, length(dateRows))
US_deaths_forecast3 <- matrix(nrow = nrow(ypost_oose3), ncol = length(dateRows))

for (k in 1:length(dateRows)) {
  US_deaths_date[k] <- us_states_ooseNA[dateRows[[k]][k], "date"]
  if (length(dateRows[[k]]) == 1) {
    US_deaths_forecast3[, k] <- ypost_oose3[, dateRows[[k]]]
  } else {
    US_deaths_forecast3[, k] <- as.numeric(rowSums(na.omit(ypost_oose3[, dateRows[[k]]])))
  }
}

USdeaths_est <- apply(US_deaths_forecast3, 2, quantile, 0.5)
USdeaths_lo <- apply(US_deaths_forecast3, 2, quantile, 0.05)
USdeaths_hi <- apply(US_deaths_forecast3, 2, quantile, 0.95)

US_deaths_forecast3Fixed <- US_deaths_forecast3

for (j in 1:ncol(US_deaths_forecast3Fixed)) {
  if (US_deathsactual$date[j] <= today() - 1) {
    US_deaths_forecast3Fixed[, j] <- US_deathsactual$US_deaths[j]
  } 
}

## Calculate cumulative deaths
US_deathscm_forecast3 <- t(apply(US_deaths_forecast3Fixed, 1, cumsum))

USdeathscm_est <- apply(US_deathscm_forecast3, 2, quantile, 0.5)
USdeathscm_lo <- apply(US_deathscm_forecast3, 2, quantile, 0.05)
USdeathscm_hi <- apply(US_deathscm_forecast3, 2, quantile, 0.95)

USdeathsDf <- US_deathsactual %>%
  mutate(deaths_est = USdeaths_est,
         deaths_lo = USdeaths_lo,
         deaths_hi = USdeaths_hi) %>%
  ## mutate(deaths_lo = ifelse(date <= today() - 1, NA, deaths_lo),
  ##        deaths_hi = ifelse(date <= today() - 1, NA, deaths_hi)) %>% 
  mutate(US_deathscm = cumsum(US_deaths)) %>% 
  mutate(deathscm_est = USdeathscm_est,
         deathscm_lo = USdeathscm_lo,
         deathscm_hi = USdeathscm_hi) %>%
  mutate(US_deaths = ifelse(date <= today() - 1, US_deaths, NA),
         US_deathscm = ifelse(date <= today() - 1, US_deathscm, NA))

###############################################################################
                                        #            USA spaghetti            #
###############################################################################

sampleQuantiles <- quantile(US_deathscm_forecast3[, ncol(US_deathscm_forecast3)],
                            seq(0.1, 0.9, length.out = 10))

Rows <- rep(NA, length(sampleQuantiles))

Dates <- USdeathsDf$date

dfList <- vector("list", length(Rows))

for (j in 1:length(Rows)) {
  idxJ <- which.min(abs(US_deathscm_forecast3[, ncol(US_deathscm_forecast3)] -
                        (sampleQuantiles[j])))


  Rows[j] <- idxJ

  dfList[[j]] <- data.frame(
    date = Dates,
    daily_deaths = US_deaths_forecast3Fixed[idxJ, ],
    cumulative_deaths= US_deathscm_forecast3[idxJ, ],
    chain = j
  )
  
}

USAspagDf <- dfList %>% plyr::rbind.fill() 

USAspagWide <- USAspagDf %>%
  pivot_wider(names_from = chain,
              values_from = c(daily_deaths, cumulative_deaths),
              names_prefix = c("chain"))

readr::write_csv(USAspagWide,
                 sprintf("../forecasts/archive/UT-COVID19-usaspaghetti-%s.csv", today()))

readr::write_csv(USAspagWide,
                 sprintf("../forecasts/UT-COVID19-usaspaghetti-latest.csv", today()))

###############################################################################
                                        #   Save forecast for US and states   #
###############################################################################

us_states_forecast_report <- us_fit_oose3_NoNA %>%
  ## mutate(deaths_hat = ifelse(date <= today() - 1, NA, deaths_hat),
  ##        deaths_lo = ifelse(date <= today() - 1, NA, deaths_lo),
  ##        deaths_hi = ifelse(date <= today() - 1, NA, deaths_hi),
  ##        deathscm_hat = ifelse(date <= today() - 2, NA, deathscm_hat),
  ##        deathscm_lo = ifelse(date <= today() - 1, NA, deathscm_lo),
  ##        deathscm_hi = ifelse(date <= today() - 1, NA, deathscm_hi)) %>% 
  dplyr::select(state,
         date,
         daily_deaths_actual = deaths,
         daily_deaths_est = deaths_hat,
         daily_deaths_90CI_lower = deaths_lo,
         daily_deaths_90CI_upper = deaths_hi,
         cumulative_deaths_actual = cumulative_deaths,
         cumulative_deaths_est = deathscm_hat,
         cumulative_deaths_90CI_lower = deathscm_lo,
         cumulative_deaths_90CI_upper = deathscm_hi,
         population = population) %>%
  left_join(statePeakDf)

readr::write_csv(us_states_forecast_report,
                 sprintf("../forecasts/archive/UT-COVID19-states-forecast-%s.csv", today()))

readr::write_csv(us_states_forecast_report,
                 sprintf("../forecasts/UT-COVID19-states-forecast-latest.csv", today()))

us_forecast_report <- USdeathsDf %>%
  mutate(
    deaths_est = ifelse(date <= today() - 1, US_deaths, deaths_est),
    deaths_lo = ifelse(date <= today() - 1, US_deaths, deaths_lo),
    deaths_hi = ifelse(date <= today() - 1, US_deaths, deaths_hi),
    ## deathscm_est = ifelse(date <= today() - 3, NA, deathscm_est),
    ## deathscm_lo = ifelse(date <= today() - 1, NA, deathscm_lo),
    ## deathscm_hi = ifelse(date <= today() - 1, NA, deathscm_hi)
  ) %>% 
  dplyr::select(date,
         daily_deaths_actual = US_deaths,
         daily_deaths_est = deaths_est,
         daily_deaths_90CI_lower = deaths_lo,
         daily_deaths_90CI_upper = deaths_hi,
         cumulative_deaths_actual = US_deathscm,
         cumulative_deaths_est = deathscm_est,
         cumulative_deaths_90CI_lower = deathscm_lo,
         cumulative_deaths_90CI_upper = deathscm_hi) %>%
  left_join(USApeakDf)

readr::write_csv(us_forecast_report,
                 sprintf("../forecasts/archive/UT-COVID19-usa-forecast-%s.csv", today()))

readr::write_csv(us_forecast_report,
                 sprintf("../forecasts/UT-COVID19-usa-forecast-latest.csv", today()))

###############################################################################
                                        #             Save output             #
###############################################################################

cat("Saving output...\n")

save(list=ls(), file=sprintf("model_fits/stan-hglm-nb-%s.Rdata", now()))

cat("Done!\n")
