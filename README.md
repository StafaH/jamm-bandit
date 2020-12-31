# JAMME-Bandit

JAMME-Bandit is a web app built for testing thompson sampling multi-armed bandits.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The app requires Python 3.X as well as [Git](https://git-scm.com/downloads) and the [Heroku CLI](https://devcenter.heroku.com/articles/getting-started-with-python#set-up) 

### Installing

After cloning the resposity you need to install the required python packages

```
pip install django
pip install gunicorn (If on Windows ignore this)
pip install django-heroku

Additional Packages:
pip install numpy
```

Detaield Instructions for how to deploy Heroku Python Apps (Free) such as this one can be found at [this link](https://devcenter.heroku.com/articles/getting-started-with-python). The free-tier can support deployment of this web app.

### Usage

There are two custom commands as part of the Django manage.py command interface
```
python manage.py populatearms 
python manage.py resetarms
```
* populatearms will take very .bmp file and create an Arm model and populate the database
* resetarms will reset the cumulative statistics on the arms for new studies

## Built With

* [Django](https://www.djangoproject.com/) - The web framework used
* [Heroku](https://devcenter.heroku.com/) - Deployment


## Authors

* **Mustafa Haiderbhai**
* **Allen Bao**
* **Emmy Liu**
* **Joe Hoang**
* **Molly Sun**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Big thanks to [Dr. Joseph Williams](http://www.josephjaywilliams.com/) for his inspiration and guidance from his course CSC2558H F 


