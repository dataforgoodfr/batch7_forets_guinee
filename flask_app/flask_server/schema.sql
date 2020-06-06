drop table if exists Post;
create table Post (
  id INTEGER NOT NULL,
  title VARCHAR(100) NOT NULL,
  date_posted DATETIME NOT NULL,
  tiff VARCHAR(20),
  msi VARCHAR(20),
  rgb VARCHAR(20),
  mask VARCHAR(20),
  infra VARCHAR(20),
  mask_msi VARCHAR(20),
  mask_rgb VARCHAR(20),
  msi_rgb VARCHAR(20),
  mask_infra VARCHAR(20),
  rgb_infra VARCHAR(20),
  msi_infra VARCHAR(20),
  mask_msi_infra VARCHAR(20),
  mask_rgb_infra VARCHAR(20),
  msi_rgb_infra VARCHAR(20),
  msi_rgb_mask VARCHAR(20),
  all_imgs VARCHAR(20),
  kpis VARCHAR(20),
  barplot VARCHAR(20),
  content TEXT NOT NULL,
  user_id INTEGER NOT NULL,
  PRIMARY KEY (id),
  FOREIGN KEY(user_id) REFERENCES user (id)
);
drop table if exists User;
create table User (
  id INTEGER NOT NULL,
	username VARCHAR(20) NOT NULL,
	email VARCHAR(120) NOT NULL,
	image_file VARCHAR(20) NOT NULL,
	password VARCHAR(60) NOT NULL,
	PRIMARY KEY (id),
	UNIQUE (username),
	UNIQUE (email)
);
