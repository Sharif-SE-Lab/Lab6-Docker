# آزمایش ششم: استفاده از Docker
## شرح آزمایش
در این آزمایش قصد داریم نحوه کار با
Docker
به عنوان یک ابزار 
Orchestration
و
Deployment
را بیاموزیم. به این منظور یک نیازمندی تعریف کرده، و برای آن یک سیستم نرم‌افزاری میکروسرویس ایجاد می‌کنیم.

## تحلیل نیازمندی‌ها
درمانگاهی یک مدل یادگیری ماشین روی داده‌های دیابت بیماران درخواست کرده است.
قرار است همه روزه نتیجه آزمایش چندین بیمار در این سیستم ثبت شود و سیستم در طول زمان به یادگیری داده‌های بیماران ادامه دهد.\
همچنین لازم است سیستم قابلیت پیشبینی نتیجه آزمایش بیمار را داشته باشد تا پزشکان درمانگاه گاها پیش از آماده شدن نتیجه اصلی آزمایش، از پیش‌بینی مدل یادگیری ماشین استفاده کنند.\
تمام ارتباطات با این سیستم باید از طریق یک
RESTfull API
محقق شود و نیازی به طراحی و ایجاد یک کارخواهِ
Front-End
برای این سیستم نیست چرا که قرار است زیرسیستم
Front-End
برون سپاری شود. سیستم باید دارای یک load-balancer معقول باشد تا در زمان‌های پیک مراجعین، دسترسی پذیری دچار آسیب نشود.
## طراحی سیستم نرم‌افزاری
به منظور ایجاد معماری سیستم از یک نگاه بالا به پایین
(Top-Down)
استفاده می‌کنیم. به این منظور ابتدا یک نمودار استقرار طراحی می‌کنیم سپس ذیل گره‌های نمودار استقرار، نمودار مولفه سیستم را طراحی خواهیم کرد.\
### نمودار استقرار
ابتدا یک گره به منظور فراهم آوردن
RESTfull API
خواسته شده قرار می‌دهیم. این گره را پشتِ
NGINX
که یک
Load-balancing WebServer
قرار می‌دهیم.
همچنین به منظور آنکه داده‌های بیماران به صورت مانا
(Persistent)
حفظ شود، گره‌ی
API
را در ارتباط با یک دیتابیس
PostgreSQL
قرار می‌دهیم.


![deployment diagram part 1](<./resources/deployment_diagram_pt1.png>)

همانطور که قابل مشاهده است از گره‌ی
API
دو نسخه به صورت همزمان مستقر شده است و تقسیم‌کننده‌ی بارِ
NGINX
با سیاست 
least_conn
بار را بین این دو نسخه تقسیم می‌کند (طبق این سیاست، هر درخواست از کارگزار، به نسخه‌ای داده می‌شود که در لحظه در حال خدمت‌رسانی به تعداد کمتری درخواست باشد). همچنین این گره‌ی
API
با استفاده از چارچوب
Django
پیاده‌سازی شده است.\
پس از آن، یک مدل یادگیری ماشین رگرسیون خطی نیاز است که داده‌های بیماران را رفته رفته یاد گرفته و برای پیش‌بینی دیابت نیز استفاده شود.


![deployment diagram part 2](./resources/deployment_diagram_pt2.png)
این مدل با پایتون ایجاد شده است و نحوه خدمت گرفتن از آن از طریق یک واسطِ پیامِ
RabbitMQ
محقق می‌شود. به این شکل گره‌ی
API
با کمترین اتصالات
(Coupling)
با مدلِ یادگیری ماشین ارتباط خواهد گرفت.

**در ادامه تصویر کلیِ نمودار استقرار را مشاهده خواهید کرد.**


![deployment diagram](./resources/deployment_diagram.jpg)

دو مولفه‌ی زیرسیستم
(Subsystem، به معنی بالا-رده ترین مولفه‌ی نرم‌افزاری که روی گره‌ی استقرار می‌شیند. شایان توجه است که به دلیل ساختار تو در تو داشتن نمودار مولفه، چنین تعریفی کاربردی است.)
که در نمودار استقرار قابل مشاهده‌اند، زیرسیستم
API
و
Model
می‌باشند که آن‌ها را در **نمودار مولفه** بررسی می‌کنیم.
### نمودار مولفه

در ادامه نمودار‌ مولفه ایجاد شده ذکر می‌شوند که درک تدقیق شده‌ای از پیاده‌سازی سیستم نرم‌افزاری به ما می‌دهد.

مولفه‌ی مدل از زیرمولفه‌های زیر تشکیل شده است.
- Regression: مدل رگرسیون که به منظور یادگیری داده‌های دیابت استفاده خواهد شد.
- DataLoader: مولفه بارگذاری داده روی مدل‌های یادگیری ماشین.
- FeatureCleanser: مولفه پاکسازی و تنظیم فیچرهای یادگیری ماشین.
- Services: مولفه‌ای که مولفه‌های دیگر را کنار هم می‌گذارد و خدمات مورد نیاز برای یادگیری و پیش‌بینی بیماران دیابتی را فراهم می‌آورد.

این مولفه دو واسط برای خدمات گیری به نام
ModelFacade
و
ModelConfigurationFacade
ارائه می‌دهد. نمای اول به منظور درخواستِ یادگیری و پیش‌بینی مدل استفاده می‌شود و نمای دوم به منظور تنظیم
Hyperparameter
های مدل استفاده خواهد شد.\
این نما
(facade)
ها با استفاده از پکیج
[nameko](https://www.nameko.io/)
ایجاد شده‌اند و استفاده از متد‌های آن‌ها در واقع با روش
RPC (Remote procedure call)
و از طریق واسطِ پیامِ
RabbitMQ
ممکن خواهد بود.

![component diagram part 1](./resources/component_diagram_pt1.png)

در ادامه مولفه‌ی
API
را خواهیم داشت که ذیل خود یک مولفه
DjangoServer
و یک مولفه‌ی
uWSGI
خواهد داشت.
مولفه‌ی
uWSGI
به عنوان
Gateway
پایتونی ما عمل خواهد کرد و درخواست‌ها و پاسخ‌های سرور را به سمت مولفه‌ی بیرونیِ
NGINX
رد می‌کند.\
همچنین مولفه‌ی
API
از طریق زیر مولفه‌ی
Model
درون خود، از واسط‌های زیرسیستم
Model (شامل مدل یادگیری ماشین)،
که این واسط‌ها همان
ModelFacade
و
ModelConfigurationFacade
هستند استفاده می‌کند.

![component diagram part 2](./resources/component_diagram_pt2.png)

**در ادامه تصویر کلیِ نمودار مولفه را مشاهده خواهید کرد.**

![component diagram](./resources/component_diagram.jpg)

## اجرای سیستم
پروژه شامل سه فایل پیکربندیِ
docker-compose
و دو فایل پیکربندی
Dockerfile
می‌باشد.
پیکربندِ اولی که راه اندازی می‌کنیم
docker-compose.yml
در کف پروژه (کنار همین فایل README) می‌باشد.
به منظور استقرار این پیکربند از دستور زیر استفاده شده است.
```bash
$ docker-compose up --build --detach
```

![first deploy](./resources/deploy/first_docker_compose.png)

در این پیکربند، گره‌های
Postgres
و
RabbitMQ
که در مولفه‌های زیر سیستم پروژه استفاده می‌شوند، به همراه شبکه‌ی داکری مستقر می‌شوند (برای شناخت گره‌های استقرار به بالاتر در این مستند و زیربخشِ نمودار استقرار مراجعه کنید).\
دستورِ
`docker ps`
که در ادامه‌ی هر استقرار استفاده می‌شود، اثبات کننده صحت مستقر بودن این نگهدارنده‌ها
(container)
می‌باشد. البته توجه شود در این مستند خیلی مواقع از دستورِ زیر بجای `docker ps` استفاده شده است:
```bash
$ docker ps --format "table {{.Image}}  {{.Ports}}  {{.Names}}" | grep "selab"
```
استفاده از تابعِ
grep
به این منظور است که تنها نگهدارنده‌های این پروژه نمایش شود (سیستمی که روی آن پروژه را مستقر می‌کنیم پروژه‌های داکری زیادی در حال اجرا دارد). پرچمِ
`format`
به منظور نمایش اندک توضیحاتی از نگهدارنده است که مورد نیاز ما است.

پیکربندِ دومی که راه اندازی می‌کنیم در پوشه‌ی مولفه‌ی
API
قرار گرفته است
(اسم پوشه: diabetes_api).
این پیکربند، مولفه‌ی گره‌های
API
و 
NGINX
را مستقر می‌سازد.
طبق پیکربندی داده شده به NGINX، این کارگزار وب روی پورت 8088 از localhost خدمت می‌رساند، و درخواست‌ها را بین دو نسخه از گره‌ی API که مستقر شده است بالانس می‌کند.

![second deploy](./resources/deploy/second_docker_compose.png)

و در آخر پیکربندِ مربوط به گره‌ی Model را مستقر می‌سازیم.
پردازه‌های حاصل از استقرار در واقع درخواست‌هایی که روی نما (facade) های مولفه‌ی ذیل این گره می‌آیند را خدمت‌رسانی می‌کنند (برای شناخت این نماها به زیربخشِ نمودار مولفه مراجعه کنید).\
این نماها با استفاده از پکیجِ nameko، درخواست‌های آمده را از روی صف‌های RabbitMQ خوانده و پاسخ را روی صف‌های دیگر و قرارداد شده از RabbitMQ می‌ریزند.

![third deploy](./resources/deploy/third_docker_compose.png)

در نهایت دستورات `docker ps` و `docker image ls` نشان دهنده‌ی درستی اجرای نگهدارنده‌ها می‌باشند.\
شایان ذکر است، که image های image های زیادی روی سیستم استقرار وجود دارد و البته database، message broker و NGINX استفاده شده پیش‌تر روی سیستم وجود داشتند.

![docker status](./resources/deploy/docker_status.png)

## پایانه‌ها (endpoints)

#### `POST` /api/data/patient/
از طریق این پایانه می‌توان اطلاعات بیمار را وارد کرد. تصویر زیر نمونه درخواست روی این پایانه و نتیجه موفقیت آمیز را نشان می‌دهد.

![post patient](./resources/api/post_patient.png)

#### `GET | PATCH | DELETE` /api/data/patient/<int:pk>
روی این پایانه می‌توان بیمار با آیدی خاص را دریافت، آپدیت یا پاک نمود.تصویر زیر نمونه درخواست روی این پایانه و نتیجه موفقیت آمیز را نشان می‌دهد.

![get patient](./resources/api/get_patient.png)


#### `PATCH` /api/data/patient/<int:pk>/predict/
روی این پایانه درخواست کرد با استفاده از مدل یادگیری ماشین، دیابت یک بیمار با آیدی خاص، پیش‌بینی شود. تصویر زیر نمونه درخواست روی این پایانه و نتیجه موفقیت آمیز را نشان می‌دهد.

![predict patient](./resources/api/predict_patient.png)

### `PATCH` /api/data/model/
می‌توان با درخواست روی این پایانه، هایپرپارامترهای مدل یادگیری ماشین را تنظیم نمود. تصویر زیر نمونه درخواست روی این پایانه و نتیجه موفقیت آمیز را نشان می‌دهد. در درخواست زیر، هایپرپارامترِ decay که نشان می‌دهد مدل پس از یادگیری هر batch چند درصد از epsilon یادگیری خود می‌کاهد.

![patch model](./resources/api/patch_model.png)


## پرسش‌ها
در ادامه به پرسش‌ها پاسخ می‌دهیم.
##### ۱. از چه نمودار/نمودارهای UML ای برای مدل‌سازی معماری MicroService خود استفاده کرده‌اید؟
طبق اصول طراحی چابک، ابتدا نمودار استقرار که معماری سختِ پروژه و سپس نمودار مولفه که معماری نرم آن را نمایش می‌دهد رسم شدند. سپس پیاده‌سازی ذیل این مدل‌ها آغاز شد.

##### ۲. مفهوم Domain-driven Design یا DDD چه ارتباطی با معماری MicroService دارد؟ در حد دو-سه خط توضیح دهید.
طراحی مبتنی بر حوزه یک روش ایجاد نرم‌افزار است که حول ایجاد با برنامه‌نویسی یک مدل دامنه که درک غنی از فرایند‌ها و قوانین حوزه دارد، تمرکز می‌کند (مارتین فولر).\
به این ترتیب با استفاده از DDD می‌توان فضای مسئله را به بخش‌های قابل درک (حتی برای مشتری) شکست و ذیل این قطعات کوچکتر پاسخِ میکروسرویس به مساله داد.

##### ۳. آیا Docker Compose یک ابزار Orchestration است؟ در حد دو-سه خط توضیح دهید.
ابزارهای ارکستر نرم‌افزاری، به ما کمک می‌کنند مولفه‌های نرم‌افزاری خود را به یکباره و بدون نیاز به قرار گرفتن روی محیط آن‌ها کنترل کنیم، تغییر اسکیل دهیم، و روی آن‌ها مانیتور داشته باشیم.\
تمام این فعالیت‌ها با استفاده از docker-compose ممکن می‌شود.
در فایل‌های پیکربندِ docker-compose چندین سرویسِ داکری قرار می‌گیرند که با استفاده از docker-compose در CLI استقرار آنها می تواند کنترل شود (مانند کاری که در این آزمایش انجام دادیم).